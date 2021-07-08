import argparse

def getargs():
    parser = argparse.ArgumentParser(description='A test program.')
    parser.add_argument('--problem',
                        type=str,
                        help='inverse problem [denoise, sr, random_mask, mask]')
    parser.add_argument('--ppar',
                        type=str,
                        help='problem parameter (not applied in denoise)')
    parser.add_argument('--snr',
                        type=int,
                        help='signal to noise ratio')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=int,
                        help='signal to noise ratio')
    parser.add_argument('--batchsize',
                        default=64,
                        type=int,
                        help='signal to noise ratio')
    parser.add_argument('--load',
                        default=False,
                        type=bool,
                        help='signal to noise ratio')
    parser.add_argument('--save',
                        default=True,
                        type=bool,
                        help='signal to noise ratio')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='signal to noise ratio')
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        help='signal to noise ratio')



    args = parser.parse_args()
    args = vars(args)

    assert args['problem']
    assert args['ppar']
    assert args['snr']

    return args