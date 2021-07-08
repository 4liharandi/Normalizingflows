import torch
import numpy as np
import data_handler

def forward_and_loss(model, left_batch, right_batch):
    z, nll = model(x=left_batch, y=right_batch)
    nll = torch.mean(nll)
    return z, nll

def compute_val_bpd(args, params, model, val_loader):
    nll = calc_val_loss(args, params, model, val_loader)
    print(f'====== In [train]: val_loss mean: {round(nll, 3)}')
    print('waiting for input')
    input()


def calc_val_loss(args, params, model, val_loader):
    print(f'In [calc_val_loss]: computing validation loss for data loader of len: {len(val_loader)} '
          f'and batch size: {params["batch_size"]}')

    with torch.no_grad():
        nlls= []
        for i_batch, batch in enumerate(val_loader):
            x, y = data_handler.extract_batches(batch, args)
            z, nll = forward_and_loss(model, x, y)
            nlls.append(nll.item())
        return np.mean(nlls)
