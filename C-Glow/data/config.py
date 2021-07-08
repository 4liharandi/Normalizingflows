import os
import flags

args = flags.getargs()

#####################
# Dataset: #
# dataset: celeba, ..
# problem: denoise, sr, mask, random_mask
#
#####################

dataset = 'celeba'
problem = args['problem']
problem_par = float(args['ppar'])
snr = args['snr']
checkpoint = args['checkpoint']
lr = args['lr']
batch_size = args['batchsize']
n_epochs = args['epochs']
n_its_per_epoch = 2**16

#################
# C-Glow-Architecture: #
#################
revnet_depth = 3
cluster_1 = 2
cluster_2 = 4
cluster_3 = 4
cluster_4 = 4
trainable_params = None
############
# Logging: #
############
show_structure = False
verbose_model = False
progress_bar = False
savemodel = args['save']
loadmodel = args['load']
shuffle_train = True
shuffle_test = False
generate_sampleimage = True

########################################################################
# These should not be changed in order avoid making program error-prone
########################################################################

###################
# Loading/saving: #
###################
folder = f'./'
modelfolder = os.path.join(folder, f'models')           # Save parameters under this name
filename = os.path.join(modelfolder, f'model.pt')
imfolder = os.path.join(folder, f'images')
params_file = os.path.join(folder, 'params.pt')
logfile = os.path.join(folder, 'hist.log')
data_folder = f'{dataset}_{problem}_{problem_par}_snr_{snr}'
train_data_targets = os.path.join(data_folder, f'train_targets.pt')
train_data_samples = os.path.join(data_folder, f'train_samples.pt')
test_data_targets = os.path.join(data_folder, f'test_targets.pt')
test_data_samples = os.path.join(data_folder, f'test_samples.pt')
datafiles = {
    'celeba':['/raid/konik/data/celeba_64_100k.npy', 64, 64, 3, 80000, True],
    'imagenet':['/raid/Amir/Projects/datasets/Tiny_imagenet.npy', 64, 64, 3, 80000, True],
    'rheo':['/raid/Amir/Projects/datasets/rheology.npy', 64, 64, 3, 1500, True],
    'chest':['/raid/Amir/Projects/datasets/X_ray_dataset_128.npy', 128, 128, 1, 80000, True]
}
num_batches = None

def get_checkpoint_path(e):
    return os.path.join(modelfolder, f'model_checkpoint_epoch_{e}.pt')

def get_abort_path():
    return os.path.join(modelfolder, f'model_abort.pt')

###################
# Setup:
###################
def setup():
    datasets=['mnist','celeba','imagenet','rheo','chest']
    problems=['denoise','sr','random_mask','mask']
    if dataset not in datasets:
        raise
    if problem not in problems:
        raise
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(imfolder):
        os.mkdir(imfolder)
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    return datafiles.get(dataset)

#########
# Data: #
#########
npy_file, img_w, img_h, channels, trainsize, channel_first = setup()
data_file = f'{problem}_{dataset}'
input_dim = channels * img_w * img_h
noise_sigma = 0
###################
# Info:
###################
def get_info(tr_par=None):
    banner = '####################################################\n'
    text = banner
    text += f'Problem: {problem}\n'
    text += f'Dataset: {dataset}\n'
    text += f'SNR: {snr}\n'
    if not noise_sigma == 0: text += f'sigma : {noise_sigma}\n'
    text += f'Problem Parameter: {problem_par}\n'
    text += f'Epochs: {n_epochs}\n'
    text += f'Batch size: {batch_size}\n'
    text += f'Batches per Epochs: {num_batches}\n'
    text += f'Learning rate: {lr}\n'
    text += f'GLOW-Blocks: {cluster_1+cluster_2+cluster_3+cluster_4}\n'
    text += f'Using pretrained model (if exists): {loadmodel}\n'
    text += f'Save model after training: {savemodel}\n'
    if checkpoint: text += f'Loading from checkpoint: {checkpoint}\n'
    if tr_par: text += f'Trainable Parameters: {tr_par}\n'
    text += banner
    if not os.path.exists(logfile) or os.path.getsize(logfile) == 0:
        log = open(logfile, 'a')
        log.write(text)
        log.write('Epoch \tTime \tNLL train\tLR\tTarget SNR\tSampled SNR\n')
        log.close()
    return text

def log(x):
    print(f"Writing log {x}")
    log = open(logfile, 'a')
    log.write(x + '\n')
    log.close()