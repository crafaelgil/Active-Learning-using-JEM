"""Training code for the joint energy-based model.

Author: Hideaki Hayashi
Ver. 1.0.0
"""

import medmnist
import utils
import torch as t
import torch.nn as nn
import torch.nn.functional as tnnF
import torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
import torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
from basenet import basenet
import json
from tqdm import tqdm
from dataload_utils import get_data
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True

import torch
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, n_classes=1):
        super(CCF, self).__init__(n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs):
    return t.FloatTensor(bs, args.n_ch, args.im_sz, args.im_sz).uniform_(-1, 1)


def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    f = model_cls(n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(
            replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[
            :, None, None, None]
        samples = choose_random * random_samples + \
            (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        if args.kl_weight > 0:
            im_noise = t.randn_like(init_sample).detach()
            x_k = t.autograd.Variable(init_sample, requires_grad=True)
            # sgld
            for k in range(n_steps):
                x_k = x_k + args.sgld_std * im_noise
                f_prime = t.autograd.grad(
                    f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
                if k == n_steps-1:
                    im_neg_orig = x_k
                    x_k = x_k + args.sgld_lr * f_prime
                    im_neg_kl = im_neg_orig
                    im_grad = t.autograd.grad(f(im_neg_kl, y=y).sum(), [
                        im_neg_kl], create_graph=True)[0]
                    im_neg_kl = im_neg_kl + args.sgld_lr * im_grad
                else:
                    x_k = x_k + args.sgld_lr * f_prime
            f.train()
            final_samples = x_k.detach()
            # update replay buffer
            if len(replay_buffer) > 0:
                replay_buffer[buffer_inds] = final_samples.cpu()
            return final_samples, im_neg_kl, im_grad
        else:
            x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(
                f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + \
                args.sgld_std * t.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples, 0, 0
    return sample_q


def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device).squeeze().long()
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)

def sqrt(x): return int(t.sqrt(t.Tensor([x])))

def plot(p, x): return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

def main(args):
    # Logging
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    # Environments
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    t.manual_seed(args.cuda_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.cuda_seed)

    # Datasets
    dload_train, dload_train_labeled, dload_train_unlabeled, dload_val, train_labeled_inds, train_unlabeled_inds = get_data(args)

    # Model
    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)

    # Optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr,
                             betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9,
                            weight_decay=args.weight_decay)

    writer = SummaryWriter()

    for al_iteration in range(10):
        print(f'Active Learning iteration #{al_iteration}')
        # Main trainig loop
        best_valid_acc = 0.0
        cur_iter = 0
        reset_decay = 1.0
        additional_step = 0
        for epoch in range(args.n_epochs):
            if epoch in args.decay_epochs:
                for param_group in optim.param_groups:
                    new_lr = param_group['lr'] * args.decay_rate
                    param_group['lr'] = new_lr
                print("Decaying lr to {}".format(new_lr))
            try:
                for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
                    if cur_iter <= args.warmup_iters:
                        lr = args.lr * reset_decay * \
                            cur_iter / float(args.warmup_iters)
                        for param_group in optim.param_groups:
                            param_group['lr'] = lr

                    x_p_d = x_p_d.to(device)
                    x_lab, y_lab = dload_train_labeled.__next__()
                    x_lab, y_lab = x_lab.to(device), y_lab.to(device).squeeze().long()

                    L = 0.
                    if args.p_x_weight > 0:  # maximize log p(x)
                        if args.class_cond_p_x_sample:
                            assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                            y_q = t.randint(0, args.n_classes,
                                            (args.batch_size,)).to(device)
                            x_q, im_neg_kl, im_grad = sample_q(
                                f, replay_buffer, y=y_q, n_steps=(args.n_steps+additional_step))
                        else:
                            # sample from log-sumexp
                            x_q, im_neg_kl, im_grad = sample_q(
                                f, replay_buffer, n_steps=(args.n_steps+additional_step))

                        fp_all = f(x_p_d)
                        fq_all = f(x_q)
                        fp = fp_all.mean()
                        fq = fq_all.mean()

                        l_p_x = -(fp - fq)
                        if cur_iter % args.print_every == 0:
                            print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                        fp - fq))
                        L += args.p_x_weight * l_p_x

                    if args.l2_weight > 0:
                        l_l2 = t.pow(fp_all, 2).mean() + t.pow(fq_all, 2).mean()
                        L += args.l2_weight * l_l2

                    if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                        logits = f.classify(x_lab)
                        l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                        if cur_iter % args.print_every == 0:
                            acc = (logits.max(1)[1] == y_lab).float().mean()
                            print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                        cur_iter,
                                                                                        l_p_y_given_x.item(),
                                                                                        acc.item()))
                        L += args.p_y_given_x_weight * l_p_y_given_x

                    if args.p_x_y_weight > 0:  # maximize log p(x, y)
                        assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                        x_q_lab, im_neg_kl, im_grad = sample_q(
                            f, replay_buffer, y=y_lab)
                        fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                        l_p_x_y = -(fp - fq)
                        if cur_iter % args.print_every == 0:
                            print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                            fp - fq))
                        L += args.p_x_y_weight * l_p_x_y

                    if args.kl_weight > 0:
                        f.requires_grad_(False)
                        loss_kl = f(im_neg_kl)
                        f.requires_grad_(True)
                        L -= args.kl_weight * loss_kl.mean()

                    # Break if the loss diverged
                    if L.abs().item() > 1e8:
                        # print("BAD BOIIIIIIIIII")
                        # 1/0
                        raise ValueError("Loss diverged.")

                    if t.isnan(L):
                        raise ValueError("Loss diverged.")
                    optim.zero_grad()
                    L.backward()
                    optim.step()
                    cur_iter += 1

                    if cur_iter % 100 == 0:
                        if args.plot_uncond:
                            if args.class_cond_p_x_sample:
                                assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                                y_q = t.randint(0, args.n_classes,
                                                (args.batch_size,)).to(device)
                                x_q, im_neg_kl, im_grad = sample_q(
                                    f, replay_buffer, y=y_q)
                            else:
                                x_q, im_neg_kl, im_grad = sample_q(
                                    f, replay_buffer)
                            plot(
                                '{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                        if args.plot_cond:  # generate class-conditional samples
                            y = t.arange(0, args.n_classes)[None].repeat(
                                args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                            x_q_y, im_neg_kl, im_grad = sample_q(
                                f, replay_buffer, y=y)
                            plot(
                                '{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

                if epoch % args.ckpt_every == 0:
                    checkpoint(f, replay_buffer, f'ckpt_alit{al_iteration}_{epoch}.pt', args, device)

                if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
                    f.eval()
                    with t.no_grad():
                        # validation set
                        correct, loss = eval_classification(f, dload_val, device)
                        print("Epoch {}: Valid Loss {}, Valid Acc {}".format(
                            epoch, loss, correct))
                        if correct > best_valid_acc:
                            best_valid_acc = correct
                            print("Best Valid!: {}".format(correct))
                            checkpoint(f, replay_buffer, f'best_valid_ckpt_alit{al_iteration}.pt', args, device)

                        print(f'epoch # {epoch + args.n_epochs * al_iteration}: {epoch} + {args.n_epochs}*{al_iteration}')

                        writer.add_scalar("acc/valid", correct, epoch + args.n_epochs * al_iteration)
                        writer.add_scalar("loss/valid", loss, epoch + args.n_epochs * al_iteration)

                    f.train()
                checkpoint(f, replay_buffer, f'last_ckpt_alit{al_iteration}.pt', args, device)

                # num_labels_to_fix = 8 # 1 per class

                # inds_to_fix = find_confidences_and_fix_unlabeled_dataset(args, f, dload_train_unlabeled, train_unlabeled_inds, device, num_labels_to_fix)

                # dload_train, dload_train_labeled, dload_train_unlabeled, dload_val, train_labeled_inds, train_unlabeled_inds = get_data(args, train_labeled_inds, train_unlabeled_inds , inds_to_fix, start_iter=False)


            except ValueError as e:
                # Reset to the best valid check point
                print(e)
                ckpt_dict = t.load(os.path.join(
                    args.save_dir, f"best_valid_ckpt_alit{al_iteration}.pt"))
                f.load_state_dict(ckpt_dict["model_state_dict"])
                replay_buffer = ckpt_dict["replay_buffer"]
                print("Reset to the best valid check point")
                reset_decay = reset_decay * 0.5
                for param_group in optim.param_groups:
                    param_group['lr'] = param_group['lr']*reset_decay
                additional_step = additional_step + 10

        num_labels_to_fix = 8 # 1 per class

        inds_to_fix = find_confidences_and_fix_unlabeled_dataset(args, f, dload_train_unlabeled, train_unlabeled_inds, device, num_labels_to_fix)

        dload_train, dload_train_labeled, dload_train_unlabeled, dload_val, train_labeled_inds, train_unlabeled_inds = get_data(args, train_labeled_inds, train_unlabeled_inds , inds_to_fix, start_iter=False)

    writer.flush()
    writer.close()

def find_confidences_and_fix_unlabeled_dataset(args, f, dload_unlabeled, train_unlabeled_inds, device, num_labels_to_fix):
    confs, confs_to_fix = [], []

    for x_p_d, y_p_d in tqdm(dload_unlabeled):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device).squeeze().long()
        logits = f.classify(x_p_d)
        with t.no_grad():
            confs.extend(nn.functional.softmax(logits, 1).cpu().numpy())
    with t.no_grad():
        confs = np.array(confs).reshape((-1, args.n_classes))

        for ind, conf in enumerate(confs):
            confs_to_fix.append((conf.max(), train_unlabeled_inds[ind]))

        confs_to_fix.sort(key=lambda x:x[0]) #Sorts by confidence for each image

        # total_num_of_unlabeled_images = len(confs_to_fix)
        # lower_n_percent = 0.1
        # num_labels_to_fix = int(len(confs_to_fix) * lower_n_percent) #Fix 10% of unlabeled images using oracle
        confs_to_fix = confs_to_fix[:num_labels_to_fix]
        inds_to_fix = [ind for conf, ind in confs_to_fix]
        inds_to_fix.sort()

        print(f'inds to fix: {inds_to_fix}')

        # print("Call oracle on " + str(num_labels_to_fix) + "/" + str(total_num_of_unlabeled_images) + " labels")
        # print("Lower " + str(int(lower_n_percent * 100)) + "% of confidences obtained: " + str(confs_to_fix))
        # print("Inds to fix: " + str(inds_to_fix))
        # print('Num inds to fix:' + str(len(inds_to_fix)))

        return inds_to_fix


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training of the JEM")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "svhn", "cifar100", "MRI", "OCT", "pathmnist", "octmnist", "pneumoniamnist", "chestmnist", "dermamnist", "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true",
                        help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    parser.add_argument("--hybrid_weight", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--opt_weight", type=float, default=0.3)
    parser.add_argument("--l2_weight", type=float, default=0.0)
    # Regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--uncond", action="store_true",
                        help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=1000,
                        help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int,
                        default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true",
                        help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true",
                        help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true",
                        help="If set, save unconditional samples")
    parser.add_argument("--n_classes", type=int, default=0)
    parser.add_argument("--semisupervision_seed", type=int, default=1234)
    parser.add_argument("--im_sz", type=int, default=32)
    parser.add_argument("--n_ch", type=int, default=3)
    parser.add_argument("--cuda_seed", type=int, default=1)
    parser.add_argument("--basenet", type=str, default="wideresnet",
                        choices=["wideresnet", "resnet18", "resnet50"])

    args = parser.parse_args()
    assert args.n_classes > 0, "Set n_classes!"
    main(args)
