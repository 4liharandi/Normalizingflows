from torch.utils import data
from torch.utils.data import DataLoader
import torch
import os

def load_sets(path):
    target_train = torch.load(os.path.join(path, 'train_targets.pt'))
    sample_train = torch.load(os.path.join(path, 'train_samples.pt'))
    target_test = torch.load(os.path.join(path, 'test_targets.pt'))
    sample_test = torch.load(os.path.join(path, 'test_samples.pt'))
    return target_train, sample_train, target_test, sample_test

def loaders(path, batch_size):
    target_train, sample_train, target_test, sample_test = load_sets(path)
    train_dataset = torch.utils.data.TensorDataset(target_train, sample_train)
    test_dataset = torch.utils.data.TensorDataset(target_test, sample_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=True)

    return train_loader, test_loader

