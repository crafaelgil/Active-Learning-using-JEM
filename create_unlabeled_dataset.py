# import medmnist
# from medmnist import INFO
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
import numpy as np
import shutil

def create_unlabeled_dataset_from(dataset, ratio):
  base_path = '/home/carlosgil/.medmnist/'

  shutil.copy2(base_path + dataset + '.npz', base_path + dataset + '-unlabeled.npz')
  unlabeled_data_set = dict(np.load(base_path + dataset + '-unlabeled.npz'))


  total_set_unlabeled_inds = []

  total_num_labels = len(unlabeled_data_set['train_labels'])

  num_classes = 8
  inds_for_each_label = [[]for _ in range(num_classes)]

  for ind in range(total_num_labels):
    label = int(unlabeled_data_set['train_labels'][ind])
    inds_for_each_label[label].append(ind)

  for label_inds in inds_for_each_label:
    num_inds_for_this_label = len(label_inds)
    num_unlabeled_imgs = int(num_inds_for_this_label * ratio)
    unlabeled_inds = np.random.randint(0, num_inds_for_this_label, num_unlabeled_imgs)

    for ind in unlabeled_inds:
      ind_in_unlabled_dataset = label_inds[ind]
      unlabeled_data_set['train_labels'][ind_in_unlabled_dataset] = 255

      total_set_unlabeled_inds.append(ind_in_unlabled_dataset)

  total_set_unlabeled_inds.sort()

  print('Unlabeled inds after creating unlabled dataset: ' + str(total_set_unlabeled_inds))
  print('Number of unlabeled inds: ' + str(len(total_set_unlabeled_inds)) + '/' + str(total_num_labels))

  np.savez(base_path + dataset + '-unlabeled.npz', **unlabeled_data_set)

create_unlabeled_dataset_from('bloodmnist', 0.1) #Take 10% of original bloodmnist dataset to be unlabeled
