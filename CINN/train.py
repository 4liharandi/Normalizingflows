from time import time
from tqdm import tqdm
import torch
from torchvision.utils import save_image
import torch.optim
import numpy as np
import os
import model
import data
import config as c
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
cinn = model.CINN()
cinn.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(cinn.optimizer, 1, gamma=0.85)
t_start = time()
nll_mean = []
target_test, image_test = next(iter(data.test_loader))

def generate_img(img, img_pt, epoch):
    save_image(img, os.path.join(c.imfolder, f'epoch_{epoch}.png'))
    torch.save(img_pt, os.path.join(c.imfolder, f'epoch_{epoch}.pt'))
    print("Image generated")


def mean(imgs):
    mean = np.zeros(imgs[0].shape)
    for i in range(len(imgs)):
        mean = np.add(mean, imgs[i])
    mean = np.divide(mean, len(imgs))
    return mean


def stack_vertically(imgs, num_show):
    for i in range(num_show):
        if i == 0:
            stack = imgs[i]
        else:
            stack = torch.cat((stack, imgs[i]), 1)
    return stack


def sample_from(num_show, num_samples, mean_samples=7):
    '''Colorize the whole test set once.
    num_show:       Numbers of different images
    num_samples:    Amount of generated images per sample. Mean of generated samples also created.
    Returns an array with
    '''
    whitespace = torch.ones((c.channels, num_show * c.img_h, 10)).cuda()
    first = True
    with torch.no_grad():
        mean = torch.zeros((c.batch_size, c.channels, c.img_w, c.img_h)).cuda()
        for j in range(mean_samples):
            z = torch.randn(c.batch_size, c.input_dim).cuda()
            sample, s_jac = cinn.reverse_sample(z.cuda(), image_test.cuda())
            if first:
                if j == 0:
                    samples = stack_vertically(sample[0:num_show], num_show)
                else:
                    if j <= num_samples: samples = torch.cat((samples, stack_vertically(sample[0:num_show], num_show)),
                                                             2)
            mean = torch.add(mean, sample)
        mean = torch.div(mean, mean_samples)
        if first:
            targets = stack_vertically(target_test[0:num_show], num_show)
            unres_images = stack_vertically(image_test[0:num_show], num_show)
            means = stack_vertically(mean[0:num_show], num_show)
            img = torch.cat((targets.cuda(), whitespace.cuda(), unres_images.cuda(), whitespace.cuda(), means.cuda(),
                             whitespace.cuda(), samples.cuda()), 2)
            img_pt = torch.cat((targets.cuda(), unres_images.cuda(), means.cuda(), samples.cuda()), 2)
        input_snr = data.compute_snr(image_test.detach().cpu().numpy(), target_test.detach().cpu().numpy())
        sampled_snr = data.compute_snr(mean.detach().cpu().numpy(), target_test.detach().cpu().numpy())
    return img, img_pt, sampled_snr, input_snr


def create_samples(epoch):
    imgs, img_pt, sampled_snr, unresolved_snr = sample_from(num_show=10, num_samples=8)
    generate_img(imgs, img_pt, epoch)
    if not c.problem == 'denoise':
        unresolved_snr = 0.0
    return sampled_snr, unresolved_snr


def print_stats(epoch, time, nll_mean, lr, unresolved_snr=None, sampled_snr=None):
    if not unresolved_snr is None:
        print(f'%.3i  \t%.2f \t%.6f \t%.2e \t%.4f \t%.4f \t({c.problem}: {c.snr})' % (epoch,
                                                                                      time,
                                                                                      nll_mean,
                                                                                      lr,
                                                                                      unresolved_snr,
                                                                                      sampled_snr), flush=True)
        c.log(
            f"{epoch} \t{time:.2f} \t{np.mean(nll_mean):.6f} \t{cinn.optimizer.param_groups[0]['lr']:.2e} \t{unresolved_snr:.4f} \t{sampled_snr:.4f}")
    else:
        print(f'%.3i  \t%.2f \t%.6f \t%.2e \t({c.problem}: {c.snr})' % (epoch,
                                                                        time,
                                                                        nll_mean,
                                                                        lr), flush=True)


def train():
    print(c.get_info(cinn.num_trainable_parameters))
    print('Epoch \tTime \tNLL train\tLR\tUnresolved SNR\tSampled SNR')
    el_time = 0
    saved_time = 0
    saved_epoch = 0

    epoch = 0
    if c.checkpoint:
        saved_epoch, saved_time = cinn.load(c.get_checkpoint_path(c.checkpoint))

    try:
        for epoch in range(saved_epoch + 1, c.n_epochs + 1):
            data_iter = iter(data.train_loader)
            nll_mean = 0
            for i_batch, data_tuple in tqdm(enumerate(data_iter),
                                            total=min(len(data.train_loader), c.n_its_per_epoch),
                                            leave=False,
                                            mininterval=1.,
                                            disable=(not c.progress_bar),
                                            ncols=83):
                cinn.optimizer.zero_grad()
                nll_mean = []
                target, image = data_tuple
                target, image = target.cuda(), image.cuda()
                z, log_det = cinn(target, image)
                nll = torch.mean(z ** 2) / 2 - torch.mean(log_det) / c.input_dim
                nll.backward()
                nll_mean.append(nll.item())
                cinn.optimizer.step()

            el_time = (time() - t_start) / 60. + saved_time
            if epoch % 5 == 0:
                sampled_snr, unresolved_snr = create_samples(epoch)
                print_stats(epoch, el_time, np.mean(nll_mean), cinn.optimizer.param_groups[0]['lr'], unresolved_snr,
                            sampled_snr)
                if c.savemodel: cinn.save(epoch, el_time, c.get_checkpoint_path(epoch))
            else:
                print_stats(epoch, el_time, np.mean(nll_mean), cinn.optimizer.param_groups[0]['lr'])
            scheduler.step()
        if c.savemodel: cinn.save(epoch, el_time, c.filename)
    except:
        print(f"Error: Experiment {c.problem}:{c.snr} was aborted")
        if c.savemodel: cinn.save(epoch, el_time, c.get_abort_path())
        raise


if __name__ == "__main__":
    train()

