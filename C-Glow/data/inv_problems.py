import config as c
import numpy as np
from torchvision import transforms
import torch

'''
problem_par in config has no effect
'''
def denoise(target):
    noise = np.random.normal(loc=0, scale=c.noise_sigma, size=np.shape(target))/np.sqrt(np.prod(np.shape(target)[1:]))
    noisy = target + noise
    return noisy

'''
problem_par in config corresponds to the down-scale factor (problem parameter = downscale factor)
'''
def sr(target):
    new_dim = int(c.img_w / c.problem_par)
    scaleop = transforms.Compose([transforms.Resize(new_dim, interpolation=transforms.InterpolationMode.NEAREST),
                                  transforms.Resize(c.img_w, interpolation=transforms.InterpolationMode.NEAREST)])
    downscaled = scaleop(torch.from_numpy(target)).cpu().detach().numpy()
    noise = np.random.normal(loc=0, scale=c.noise_sigma, size=np.shape(target))/np.sqrt(np.prod(np.shape(target)[1:]))
    downscaled = downscaled + noise
    return downscaled


'''
problem_par in config corresponds to the probability of keeping image information (problem parameter = keep probability)
'''
def random_mask(target):
    mask = (np.random.uniform(size = np.shape(target)) < c.problem_par).astype(np.float32)
    masked = target * mask
    noise = np.random.normal(loc=0, scale=c.noise_sigma, size=np.shape(target))/np.sqrt(np.prod(np.shape(target)[1:]))
    masked = masked + noise
    return masked

'''
problem_par in config corresponds to the mask size (problem parameter = mask size)
'''
def mask(target):
    b_size, channels, img_w, img_h = np.shape(target)
    mask = np.ones([b_size, channels, img_w, img_h], dtype=np.float32)
    for i in range(b_size):
        msk_w = np.random.randint(int(c.problem_par) // 2, img_w - int(c.problem_par) // 2)
        msk_h = np.random.randint(int(c.problem_par) // 2, img_h - int(c.problem_par) // 2)
        start_w, end_w = msk_w - int(c.problem_par) // 2, msk_w + int(c.problem_par) // 2
        start_h, end_h = msk_h - int(c.problem_par) // 2, msk_h + int(c.problem_par) // 2
        mask[i,:,start_w:end_w,start_h:end_h] *= 0.0
    masked = target * mask
    noise = np.random.normal(loc=0, scale=c.noise_sigma, size=np.shape(target))/np.sqrt(np.prod(np.shape(target)[1:]))
    masked = masked + noise
    return masked