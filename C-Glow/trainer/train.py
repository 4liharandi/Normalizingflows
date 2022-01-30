from torchvision import utils
import time
import sys
import math
from globals import device
import helper
import os
from .loss import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader

def init_train_configs(args):
    train_configs = {'reg_factor': args.reg_factor}  # lambda
    print(f'In [init_train_configs]: \ntrain_configs: {train_configs}\n')
    return train_configs

def log(logfile, epoch, optim_step, time, nll, lr, PSNR):
    x = f'{epoch} \t{optim_step} \t{time} \t{nll} \t{lr} \t{PSNR}\n'
    log = open(logfile, 'a')
    log.write(x + '\n')
    log.close()

def print_info(logfile, epoch, optim_step, time, nll, lr, PSNR=None, log_info=False):
    str = f'{epoch} \t{optim_step} \t{time} \t{nll} \t{lr}'
    if PSNR:
        str += f' \t{PSNR}'
    print(str)
    if log_info:
        log(logfile, epoch, optim_step, time, nll, lr, PSNR)

def adjust_lr(current_lr, initial_lr, step, epoch_steps):
    curr_epoch = math.ceil(step / epoch_steps)  # epoch_steps is the number of steps to complete an epoch
    threshold = 100  # linearly decay after threshold
    if curr_epoch > threshold:
        extra_epochs = curr_epoch - threshold
        decay = initial_lr * (extra_epochs / threshold)
        current_lr = initial_lr - decay

    print(f'In [adjust_lr]: step: {step}, curr_epoch: {curr_epoch}, lr: {current_lr}')
    return current_lr, curr_epoch

def compute_peak_snr(target, sample):
    def MSE(target, sample):
        assert target.shape == sample.shape
        batch, channels, N1, N2 = target.shape
        return np.mean(np.sum((target - sample), axis=(1, 2, 3)) ** 2 / (channels * N1 * N2))
    target, sample = target.detach().cpu().numpy(), sample.detach().cpu().numpy()
    return 20 * np.log10(np.max(target)) - 10 * np.log10(MSE(target, sample))

def data_normalization(x):
    
    #x = x.astype('float32')
    x = x - x.min()
    x /= (x.max())
    
    return x

def train(args, params, train_configs, model, optimizer, current_lr, resume=False, last_optim_step=0):
    # getting data loaders

    data = np.load('/raid/Amir/Projects/datasets/CT_dataset/horizontal_snr25.0.npz')

    train_images = data['x_train']
    test_images = data['x_test']
    
    y_train = data['y_train']
    y_test = data['y_test']

    #train_images = np.load('/raid/Amir/Projects/datasets/CT_dataset/train_gt_horizontal_snr40.npy')
    #print(np.shape(train_images))
    train_images = data_normalization(train_images)
    train_images = torch.from_numpy(train_images).permute(0, 3, 1, 2).cpu().detach().numpy()
    #print(np.shape(train_images))
    train_images = torch.Tensor(train_images)

    #y_train = np.load('/raid/Amir/Projects/datasets/CT_dataset/train_fbp_horizontal_snr40.npy')
    #print(np.shape(y_train))
    y_train = data_normalization(y_train)
    y_train = torch.from_numpy(y_train).permute(0, 3, 1, 2).cpu().detach().numpy()
    y_train = torch.Tensor(y_train)
    #test_images = np.load('/raid/Amir/Projects/datasets/CT_dataset/test_gt_horizontal_snr40.npy')
    test_images = data_normalization(test_images)
    test_images = torch.from_numpy(test_images).permute(0, 3, 1, 2).cpu().detach().numpy()
    test_images = torch.Tensor(test_images)
    #y_test = np.load('/raid/Amir/Projects/datasets/CT_dataset/test_fbp_horizontal_snr40.npy')
    y_test = data_normalization(y_test)
    y_test = torch.from_numpy(y_test).permute(0, 3, 1, 2).cpu().detach().numpy()
    y_test = torch.Tensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(train_images,y_train)
    test_dataset = torch.utils.data.TensorDataset(test_images,y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=True)




    #train_loader, val_loader = data_handler.init_data_loaders(args, params)
    
    val_labels, val_imgs = next(iter(val_loader))
    val_labels, val_imgs = val_labels.cuda(), val_imgs.cuda()
    # adjusting optim step
    optim_step = last_optim_step + 1 if resume else 1
    max_optim_steps = params['iter']
    paths = helper.compute_paths(args, params)
    log_file = paths['log_file']
    print(params['checkpoint_freq'])
    if resume:
        print(f'In [train]: resuming training from optim_step={optim_step} - max_step: {max_optim_steps}')
    begin_time = time.time()
    epoch_tmp = 0
    nll_sum = 0
    optims_in_epoch = 0
    # optimization loop
    while optim_step < max_optim_steps:
        # after each epoch, adjust learning rate accordingly
        current_lr, current_epoch = adjust_lr(current_lr, initial_lr=params['lr'], step=optim_step, epoch_steps=len(train_loader))  # now only supports with batch size 1
        if current_epoch > 300:
            exit(0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f'In [train]: optimizer learning rate adjusted to: {current_lr}\n')
        sampled = False
        for i_batch, batch in enumerate(train_loader):
            if optim_step > max_optim_steps:
                print(f'In [train]: reaching max_step or lr is zero. Terminating...')
                return  # ============ terminate training if max steps reached
            # forward pass
            target, image = data_handler.extract_batches(batch, args)
            target, image = target.cuda(), image.cuda()
            z, nll = forward_and_loss(model, target, image)
            # regularize left loss
            if train_configs['reg_factor'] is not None:
                loss = train_configs['reg_factor'] * nll
            else:
                loss = nll

            # backward pass and optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()
            optims_in_epoch += 1
            nll_sum += nll.item()
            end_time = time.time()

            # saving checkpoint
            if (optim_step > 0 and optim_step % params['checkpoint_freq'] == 0) or current_lr == 0:
                checkpoints_path = paths['checkpoints_path']
                helper.make_dir_if_not_exists(checkpoints_path)
                helper.save_checkpoint(checkpoints_path, optim_step, model, optimizer, loss, current_lr)
                print("In [train]: Checkpoint saved at iteration", optim_step)

            if epoch_tmp != current_epoch:
                elapsed = end_time - begin_time
                epoch_tmp = current_epoch
                mean_nll = nll_sum / optims_in_epoch
                nll_sum = 0
                optims_in_epoch = 0
                if (current_epoch % 10 == 0) or current_lr == 0 and not sampled:
                    checkpoints_path = paths['checkpoints_path']
                    helper.make_dir_if_not_exists(checkpoints_path)
                    helper.save_checkpoint(checkpoints_path, optim_step, model, optimizer, loss, current_lr, epoch=True,
                                           epoch_num=current_epoch)
                    sampled = True
                    samples_path = paths['samples_path']
                    helper.make_dir_if_not_exists(samples_path)
                    sampled_images, _ = model(x=val_imgs, reverse=True)
                    utils.save_image(sampled_images, f'{samples_path}/epoch_{str(current_epoch).zfill(4)}.png', nrow=10)
                    PSNR = compute_peak_snr(val_labels, sampled_images)
                    print_info(log_file, current_epoch, optim_step, elapsed, mean_nll, current_lr, PSNR, log_info=True)
                    print(f'\nIn [train]: Sample saved at iteration {optim_step} to: \n"{samples_path}"\n')
                    exp_dir = helper.compute_paths(args, params)['exp_base']
                    log = open(os.path.join(exp_dir, 'psnr.log'), 'a')
                    sample_from( exp_dir, log, current_epoch,10,26, model,val_imgs, val_labels, mean_samples=25)
                    log.close()

                else:
                    print_info(log_file, current_epoch, optim_step, elapsed, mean_nll, current_lr)

            optim_step += 1
            end_time = time.time()

            if current_lr == 0:
                print('In [train]: current_lr = 0, terminating the training...')
                sys.exit(0)

def sample(args, params, model, optimizer):
    
    #train_loader1, val_loader1 = data_handler.init_data_loaders(args, params)
    #val_labels1, val_imgs1 = next(iter(val_loader1))

    #print(torch.min(val_labels1[1,:,:,1]))

    
    in_data = np.load('/raid/Amir/Projects/conditional_Trumpet/datasets/measure_de.npy')
    in_data =  torch.from_numpy(in_data).permute(0, 3, 1, 2).cpu().detach().numpy()
    in_data = torch.Tensor(in_data)
    out_data = np.load('/raid/Amir/Projects/conditional_Trumpet/datasets/gt_de.npy')
    out_data =  torch.from_numpy(out_data).permute(0, 3, 1, 2).cpu().detach().numpy()
    out_data = torch.Tensor(out_data)

    test_dataset = torch.utils.data.TensorDataset(out_data, in_data)

    val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=True)

    val_labels, val_imgs = next(iter(val_loader))
    val_labels = torch.div(val_labels, 255)
    val_imgs = torch.div(val_imgs, 255)


    #print(torch.min(val_labels[1,:,:,1]))


    val_labels, val_imgs = val_labels.cuda(), val_imgs.cuda()
    exp_dir = helper.compute_paths(args, params)['exp_base']
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    for e in range(200, 201, 10):
        log = open(os.path.join(exp_dir, 'psnr.log'), 'a')
        cp = os.path.join(checkpoint_dir, f'epoch={e}.pt')
        sample_from(cp, exp_dir, log, e, 10, 26, model, val_imgs, val_labels, optimizer,  mean_samples=25)
        log.close()
        print(f'epoch {e} generated')

def sample_from(img_dir, log, epoch,num_show, num_samples, model, val_imgs, val_labels , mean_samples=25):
	
	#checkpoint = torch.load(path,map_location = device)
	#print(f'loaded epoch {epoch}')
	#model.load_state_dict(checkpoint['model_state_dict'])

	whitespace = torch.ones((1, num_show * 64, 10)).cuda()

	first = True

	with torch.no_grad():
		mean = torch.zeros((256, 1, 64, 64)).cuda()
		for j in range(mean_samples):
			sample, _ = model(x=val_imgs, reverse=True, eps_std=0.2)

			if first:
				if j == 0:

					samples = stack_vertically(sample[0:num_show], num_show)
				else:

					if j <= num_samples: samples = torch.cat((samples, stack_vertically(sample[0:num_show], num_show)),2)

			mean = torch.add(mean, sample)
		mean = torch.div(mean, mean_samples)
		if first:
			targets = stack_vertically(val_labels[0:num_show], num_show)
			unres_images = stack_vertically(val_imgs[0:num_show], num_show)
			means = stack_vertically(mean[0:num_show], num_show)
			img = torch.cat((targets.cuda(), whitespace.cuda(), unres_images.cuda(), whitespace.cuda(), means.cuda(),whitespace.cuda(), samples.cuda()), 2)
		sampled_snr = compute_peak_snr(val_labels[5:num_show], mean[5:num_show])
		log.write(f'{epoch} \t{sampled_snr}\n')
		
		sam25_dir = os.path.join(img_dir, 'ehsan')
		if not os.path.exists(sam25_dir):
			os.mkdir(sam25_dir)
		#np.save(os.path.join(sam25_dir, 'final.npy'), final)
		save_image(img, os.path.join(sam25_dir, f'epoch_{epoch}.png'))

        #np.save(os.path.join(sam25_dir, f'mean_epoch_{epoch}.npy'), mean_pt)
        #np.save(os.path.join(sam25_dir, f'target_epoch_{epoch}.npy'), target_pt)

def stack_vertically(imgs, num_show):
    for i in range(num_show):
        if i == 0:
            stack = imgs[i]
        else:
            stack = torch.cat((stack, imgs[i]), 1)
    return stack
