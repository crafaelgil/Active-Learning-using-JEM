"""Test code for the joint energy-based model.

Author: Hideaki Hayashi
Ver. 1.0.0
"""

import medmnist
import utils
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
import torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
from basenet import basenet
import pdb
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from tqdm import tqdm
from dataload_utils import get_test_data
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True


class F(nn.Module):
    def __init__(self, n_classes=0):
        super(F, self).__init__()
        self.f = basenet(args.basenet, n_ch=args.n_ch)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, n_classes=0):
        super(CCF, self).__init__(n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def init_random(bs):
    return t.FloatTensor(bs, 3, 32, 32).uniform_(-1, 1)


def sample_p_0(device, replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs), []
    buffer_size = len(replay_buffer) if y is None else len(
        replay_buffer) // n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs)
    choose_random = (t.rand(bs) < args.reinit_freq).float()[
        :, None, None, None]
    samples = choose_random * random_samples + \
        (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(args, device, f, replay_buffer, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(device, replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(args.n_steps):
        f_prime = t.autograd.grad(
            f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def uncond_samples(f, args, device, save=True):
    def sqrt(x): return int(t.sqrt(t.Tensor([x])))
    def plot(p, x): return tv.utils.save_image(
        t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    replay_buffer = t.FloatTensor(args.buffer_size, 3, 32, 32).uniform_(-1, 1)
    for i in range(args.n_sample_steps):
        samples = sample_q(args, device, f, replay_buffer)
        if i % args.print_every == 0 and save:
            plot('{}/samples_{}.png'.format(args.save_dir, i), samples)
        print(i)
    return replay_buffer


def cond_samples(f, replay_buffer, args, device, fresh=False):
    def sqrt(x): return int(t.sqrt(t.Tensor([x])))
    def plot(p, x): return tv.utils.save_image(
        t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    if fresh:
        replay_buffer = uncond_samples(f, args, device, save=False)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[i * 100: (i + 1) * 100].to(device)
        y = f.classify(x).max(1)[1]
        all_y.append(y)

    all_y = t.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    for i in range(100):
        this_im = []
        for l in range(10):
            this_l = each_class[l][i * 10: (i + 1) * 10]
            this_im.append(this_l)
        this_im = t.cat(this_im, 0)
        if this_im.size(0) > 0:
            plot('{}/samples_{}.png'.format(args.save_dir, i), this_im)
        print(i)


def test_clf(f, args, device):
    dload = get_test_data(args)

    def sample(x, n_steps=args.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(
                f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    corrects, losses, pys, preds = [], [], [], []
    gts, confs = [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device).squeeze().long()
        if args.n_steps > 0:
            x_p_d = sample(x_p_d)
        logits = f.classify(x_p_d)
        py = nn.Softmax()(f.classify(x_p_d)).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduce=False)(
            logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())
        with t.no_grad():
            gts.extend(y_p_d.cpu().numpy())
            confs.extend(nn.functional.softmax(logits, 1).cpu().numpy())
    with t.no_grad():
        loss = np.mean(losses)
        correct = np.mean(corrects)
        t.save({"losses": losses, "corrects": corrects, "pys": pys},
               os.path.join(args.save_dir, "vals.pt"))
        print("Loss={}".format(loss))
        print("Accuracy={}".format(correct))
        ece = ECE(10)
        confs = np.array(confs).reshape((-1, args.n_classes))
        gts = np.array(gts)
        calibration_score = ece.measure(confs, gts)
        print("ECE={}".format(calibration_score))
        diagram = ReliabilityDiagram(10)
        pl = diagram.plot(confs, gts)
        pl.savefig(f"{args.save_dir}/confidence.png")


def main(args):
    # Logging
    utils.makedirs(args.save_dir)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    # Environments
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    t.manual_seed(args.cuda_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.cuda_seed)

    # Model    
    model_cls = F if args.uncond else CCF
    f = model_cls(n_classes=args.n_classes)
    print(f"loading model from {args.load_path}")
    ckpt_dict = t.load(args.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]
    f = f.to(device)

    # Test
    test_clf(f, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test of the JEM")
    parser.add_argument("--dataset", default="cifar_test", type=str, choices=[
                        "cifar_train", "cifar_test", "svhn_test", "svhn_train", "MRI", "OCT", "pathmnist", "octmnist", "pneumoniamnist", "chestmnist", "dermamnist", "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"], help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--n_classes", type=int, default=0)
    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=3e-2)
    # network
    parser.add_argument("--norm", type=str, default=None,
                        choices=[None, "norm", "batch", "instance", "layer", "act"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str,
                        default=None)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true")
    parser.add_argument("--fresh_samples", action="store_true",
                        help="If set, then we generate a new replay buffer from scratch for conditional sampling,"
                             "Will be much slower.")
    parser.add_argument("--im_sz", type=int)
    parser.add_argument("--n_ch", type=int)
    parser.add_argument("--cuda_seed", type=int, default=1)
    parser.add_argument("--basenet", type=str, default="wideresnet",
                        choices=["wideresnet", "resnet18", "resnet50"])
    args = parser.parse_args()
    assert args.n_classes > 0, "Set n_classes!"
    main(args)
