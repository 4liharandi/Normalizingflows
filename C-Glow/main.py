from helper import read_params
import trainer
import helper
import models
import argparse
from torch import optim



'''
python3 main.py --model improved_so_large_longer --img_size 512 1024 --dataset cityscapes --direction label2photo 
--n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint
'''

def     read_params_and_args():
    parser = argparse.ArgumentParser(description='Glow trainer')
    parser.add_argument('--dataset', type=str, help='the name of the dataset')
    parser.add_argument('--use_comet', action='store_true')
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--use_bmaps', action='store_true')  # using boundary maps
    parser.add_argument('--do_ceil', action='store_true')  # flooring b_maps
    parser.add_argument('--do_lu', action='store_true')
    parser.add_argument('--grad_checkpoint', action='store_true')
    parser.add_argument('--limited', action='store_true')  # limit dataset for debugging

    parser.add_argument('--local', action='store_true')  # local experiments
    parser.add_argument('--run', type=str)
    parser.add_argument('--image_name', type=str)
    # parser.add_argument('--transfer', action='store_true')  # content transfer local
    # parser.add_argument('--sketch2photo', action='store_true')

    parser.add_argument('--last_optim_step', type=int)
    parser.add_argument('--sample_freq', type=int)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--val_freq', type=int)
    parser.add_argument('--prev_exp_id', type=str)
    parser.add_argument('--max_step', type=int)
    parser.add_argument('--n_samples', type=int)

    parser.add_argument('--n_block', type=int)
    parser.add_argument('--n_flow', nargs='+', type=int)
    parser.add_argument('--img_size', nargs='+', type=int)  # in height width order: e.g. --img_size 128 256
    parser.add_argument('--bsize', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--temp', type=float)  # temperature
    parser.add_argument('--reg_factor', type=float)

    # Note: left_lr is str since it is used only for finding the checkpoints path of the left glow
    parser.add_argument('--left_step', type=int)  # left glow optim_step (pretrained) in c_flow
    parser.add_argument('--left_lr', type=str)  # the lr using which the left glow was trained
    parser.add_argument('--left_pretrained', action='store_true')  # use pre-trained left glow inf c_flow
    parser.add_argument('--w_conditional', action='store_true')
    parser.add_argument('--act_conditional', action='store_true')
    parser.add_argument('--coupling_cond_net', action='store_true')
    parser.add_argument('--no_validation', action='store_true')
    parser.add_argument('--investigate_dual_glow', action='store_true')
    parser.add_argument('--sample', action='store_true')

    # not used anymore
    parser.add_argument('--left_cond', type=str)  # condition used for training left glow, if any

    # args for Cityscapes
    parser.add_argument('--model', type=str, default='glow', help='which model to be used: glow, c_flow, ...')
    # parser.add_argument('--cond_mode', type=str, help='the type of conditioning in Cityscapes')
    parser.add_argument('--direction', type=str, default='label2photo')

    parser.add_argument('--train_on_segment', action='store_true')  # train/synthesis with vanilla Glow on segmentations
    parser.add_argument('--sanity_check', action='store_true')
    parser.add_argument('--test_invertibility', action='store_true')

    # data preparation
    parser.add_argument('--create_boundaries', action='store_true')
    parser.add_argument('--create_tf_records', action='store_true')

    # evaluation
    parser.add_argument('--infer_on_set', action='store_true')
    parser.add_argument('--resize_for_fcn', action='store_true')
    parser.add_argument('--infer_and_evaluate_c_glow', action='store_true')
    parser.add_argument('--infer_and_evaluate_dual_glow', action='store_true')
    parser.add_argument('--infer_dual_glow', action='store_true')
    parser.add_argument('--sampling_round', type=int)
    parser.add_argument('--all_sampling_rounds', action='store_true')
    parser.add_argument('--split_set', type=str)

    parser.add_argument('--eval_complete', action='store_true')
    parser.add_argument('--eval_fcn', action='store_true')
    parser.add_argument('--eval_ssim', action='store_true')
    parser.add_argument('--gt', action='store_true')  # used only as --evaluate --gt for evaluating ground-truth images

    # args mainly for the experiment mode: --exp should be used for the following (should be revised though)
    parser.add_argument('--random_samples', action='store_true')
    parser.add_argument('--new_condition', action='store_true')
    parser.add_argument('--seg_image', type=str)  # name of the segmentation used for random sampling

    parser.add_argument('--compute_val_bpd', action='store_true')
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--interp', action='store_true')
    parser.add_argument('--new_cond', action='store_true')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--sample_c_flow', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--syn_segs', action='store_true')  # segmentation synthesis for c_flow
    parser.add_argument('--trials', type=int)

    arguments = parser.parse_args()
    parameters = read_params('params.json')[arguments.dataset]  # parameters related to the wanted dataset
    return arguments, parameters


def adjust_params(args, params):
    """
    Change the default params if specified in the arguments.
    :param args:
    :param params:
    :return:
    """
    if args.n_block is not None:
        params['n_block'] = args.n_block

    if args.n_flow is not None:
        params['n_flow'] = args.n_flow

    if args.bsize is not None:
        params['batch_size'] = args.bsize

    if args.lr is not None:
        params['lr'] = args.lr

    if args.temp is not None:
        params['temperature'] = args.temp

    if args.img_size is not None:
        params['img_size'] = args.img_size

    if args.sample_freq is not None:
        params['sample_freq'] = args.sample_freq

    if args.checkpoint_freq is not None:
        params['checkpoint_freq'] = args.checkpoint_freq

    if args.val_freq is not None:
        params['val_freq'] = args.val_freq

    if args.no_validation:
        params['monitor_val'] = False

    if args.max_step:
        params['iter'] = args.max_step

    if args.n_samples:
        params['n_samples'] = args.n_samples

    print('In [adjust_params]: params adjusted')
    return params

def main():
    args, params = read_params_and_args()
    params = adjust_params(args, params)

    train_configs = trainer.init_train_configs(args)
    model = models.interface.init_c_glow(args, params)
    model.cuda()
    header = helper.print_info(args, params, model=model, which_info='all')
    helper.compute_paths(args, params, header=header)
    optimizer = optim.Adam(model.parameters())
    if args.sample:
        trainer.sample(args, params, model, optimizer)
    elif args.resume_train:
        optim_step = args.last_optim_step
        checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
        model, optimizer, _, lr = helper.load_checkpoint(checkpoints_path, optim_step, model, optimizer)
        lr = params['lr']
        trainer.train(args, params, train_configs, model, optimizer, lr, resume=True, last_optim_step=optim_step)
    else:
        lr = params['lr']
        trainer.train(args, params, train_configs, model, optimizer, lr)
    '''
    elif args.exp and (args.eval_fcn or args.eval_ssim or args.eval_complete):
    run_evaluation(args, params)
    '''

if __name__ == '__main__':
    main()