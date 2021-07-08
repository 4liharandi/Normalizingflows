from os.path import join, exists
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config as c
from tqdm import tqdm
import inv_problems as p
import matplotlib.pyplot as plt


def generate_sigma(target):
    return 10 ** (-c.snr / 20.0) * np.sqrt(np.mean(np.sum(np.square(np.reshape(target, (np.shape(target)[0], -1))), -1)))

def compute_snr(sample, target):
    n_p = np.sum(np.square(np.subtract(target, sample)), axis=(1, 2, 3))
    s_p = np.sum(np.square(target), axis=(1, 2, 3))
    SNR = 10 * np.log10(np.mean(s_p / n_p))
    return SNR

def make_sample_images(target_train, sample_train, target_test, sample_test):
    rows = []
    count = 0

    for i in range(10):
        row = torch.cat((target_train[count], sample_train[count]), 2)
        for j in range(10):
            count += 1
            row = torch.cat((row, target_train[count], sample_train[count]),2)
        rows.append(row.cpu().numpy())
    count=0
    for i in range(10):
        row = torch.cat((target_test[count], sample_test[count]), 2)
        for j in range(10):
            count += 1
            row = torch.cat((row, target_test[count], sample_test[count]),2)
        rows.append(row.cpu().numpy())
    imgs = np.clip(np.rollaxis(np.concatenate(rows, axis=1), 0, 3), 0, 1)
    plt.axis('off')
    plt.imshow(imgs)
    plt.savefig(join(c.imfolder, f'samples_{c.dataset}_{c.snr}_{c.problem}_{c.problem_par}.png'), bbox_inches='tight',dpi=600)

def get_function():
    if c.problem == 'denoise':
        return p.denoise
    elif c.problem == 'sr':
        return p.sr
    elif c.problem == 'random_mask':
        return p.random_mask
    elif c.problem == 'mask':
        return p.mask
    else:
        raise

def save_sets(target_train, sample_train, target_test, sample_test):
    torch.save(target_train, c.train_data_targets)
    torch.save(sample_train, c.train_data_samples)
    torch.save(target_test, c.test_data_targets)
    torch.save(sample_test, c.test_data_samples)

def load_sets():
    target_train = torch.load(c.train_data_targets)
    sample_train = torch.load(c.train_data_samples)
    target_test = torch.load(c.test_data_targets)
    sample_test = torch.load(c.test_data_samples)
    return target_train, sample_train, target_test, sample_test

def create_samples():
    data = np.load(c.npy_file)
    if c.channel_first:
        data = torch.from_numpy(data).permute(0, 3, 1, 2).cpu().detach().numpy()
    targets = []
    samples = []
    function = get_function()
    c.noise_sigma = generate_sigma(data)
    sample = function(data)
    for i in tqdm(range(data.shape[0])):
        targets.append(data[i])
        samples.append(sample[i])
    imgs_targets = torch.Tensor(np.stack(targets, axis=0))
    imgs_samples = torch.Tensor(np.stack(samples, axis=0))
    return imgs_targets[0:c.trainsize], imgs_samples[0:c.trainsize], imgs_targets[c.trainsize:], imgs_samples[c.trainsize:]

def prepare_sets():
    if not (exists(c.train_data_targets) and exists(c.train_data_samples) and exists(c.test_data_targets) and exists(c.test_data_samples)):
        print(f"Generating train and test sets with {c.dataset} for {c.problem} with snr={c.snr} and parameter={c.problem_par}...")
        target_train, sample_train, target_test, sample_test = create_samples()
        save_sets(target_train, sample_train, target_test, sample_test)
    else:
        print(f"Loading train and test sets with {c.dataset} for {c.problem} with snr={c.snr} and parameter={c.problem_par}...")
        target_train, sample_train, target_test, sample_test = load_sets()

    train_dataset = torch.utils.data.TensorDataset(target_train, sample_train)
    test_dataset = torch.utils.data.TensorDataset(target_test, sample_test)

    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=c.shuffle_train, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=c.batch_size, shuffle=c.shuffle_test, num_workers=4, pin_memory=True, drop_last=True)

    if c.generate_sampleimage: make_sample_images(target_train, sample_train, target_test, sample_test)
    c.num_batches = len(train_loader)
    return train_loader, test_loader

train_loader, test_loader = prepare_sets()

if __name__ == "__main__":
    t_exp_train, i_exp_train = next(iter(train_loader))
    t_exp_test, i_exp_test = next(iter(test_loader))
    torch.save(t_exp_train, os.path.join(c.folder, f't_exp_train_{c.problem}_{c.snr}.pt'))
    torch.save(i_exp_train, os.path.join(c.folder, f'i_exp_train_{c.problem}_{c.snr}.pt'))
    torch.save(t_exp_test, os.path.join(c.folder, f't_exp_test_{c.problem}_{c.snr}.pt'))
    torch.save(i_exp_test, os.path.join(c.folder, f'i_exp_test_{c.problem}_{c.snr}.pt'))