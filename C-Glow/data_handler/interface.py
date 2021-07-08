from data_handler.loader import loaders
from globals import maps_fixed_conds, device

def extract_batches(batch, args):
    """
    This function depends onf the dataset and direction.
    :param batch:
    :param args:
    :return:
    """
    datas = ['denoisemin3','denoisemin1','denoise1','denoise5','randommask','mask','superresolution4','superresolution8']
    if args.dataset in datas:
        target = batch[0]
        sample = batch[1]
        left_batch = sample
        right_batch = target
    else:
        raise NotImplementedError
    return left_batch, right_batch


def init_data_loaders(args, params):
    batch_size = params['batch_size']
    if args.dataset == 'denoisemin3':
        train_loader, val_loader = loaders('./data/celeba_denoise_0.0_snr_-3', batch_size)
    elif args.dataset == 'denoisemin1':
        train_loader, val_loader = loaders('./data/celeba_denoise_0.0_snr_-1', batch_size)
    elif args.dataset == 'denoise1':
        train_loader, val_loader = loaders('./data/celeba_denoise_0.0_snr_1', batch_size)
    elif args.dataset == 'denoise5':
        train_loader, val_loader = loaders('./data/celeba_denoise_0.0_snr_5', batch_size)
    elif args.dataset == 'randommask':
        train_loader, val_loader = loaders('./data/celeba_random_mask_0.2_snr_10', batch_size)
    elif args.dataset == 'mask':
        train_loader, val_loader = loaders('./data/celeba_mask_15.0_snr_10', batch_size)
    elif args.dataset == 'superresolution4':
        train_loader, val_loader = loaders('./data/celeba_sr_4.0_snr_10', batch_size)
    elif args.dataset == 'superresolution8':
        train_loader, val_loader = loaders('./data/celeba_sr_8.0_snr_10', batch_size)
    else:
        raise NotImplementedError

    print(f'\nIn [init_data_loaders]: training with data loaders of size: \n'
          f'train_loader: {len(train_loader):,} \n'
          f'val_loader: {len(val_loader):,} \n'
          f'and batch_size of: {batch_size}\n')
    return train_loader, val_loader
