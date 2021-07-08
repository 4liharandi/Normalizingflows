import glob
import os


def compute_paths(args, params, additional_info=None, header=None):
    assert args.model != 'glow' and additional_info is None  # these cases not implemented yet

    basedir = './output'
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    # explicit experiment dir
    dataset = args.dataset
    snr = params['snr']
    ppar = params['ppar']
    structure = params['n_flow']
    struc_str = ""
    for s in structure:
        struc_str += f'{s}_'
    exp_base = os.path.join(basedir, f'{struc_str}{dataset}_{snr}_{ppar}')
    if not os.path.exists(exp_base):
        os.mkdir(exp_base)

    # samples path for train
    samples_path = os.path.join(exp_base, 'samples')
    if not os.path.exists(exp_base):
        os.mkdir(exp_base)

    # generic infer path
    infer_path = os.path.join(exp_base, 'eval')
    if not os.path.exists(infer_path):
        os.mkdir(infer_path)

    # path for evaluation
    eval_path = os.path.join(infer_path, 'eval')
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    val_path = os.path.join(eval_path, 'val_imgs')
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    # where inferred val images are stored inside the eval folder
    train_vis_path = os.path.join(eval_path, 'train_vis')
    if not os.path.exists(train_vis_path):
        os.mkdir(train_vis_path)

    diverse_path = os.path.join(infer_path, 'diverse')
    if not os.path.exists(diverse_path):
        os.mkdir(diverse_path)

    # checkpoints path
    checkpoints_path = os.path.join(exp_base, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    # log-path
    log_file_path = os.path.join(exp_base, 'hist.log')
    if header:
        if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
            log = open(log_file_path, 'a')
            log.write(header)
            log.write('Epoch\tOptim Step \tTime \tNLL train \tLR \tPSNR\n')
            log.close()

    return {
        'samples_path': samples_path,
        'eval_path': eval_path,
        'val_path': val_path,
        'train_vis': train_vis_path,
        'checkpoints_path': checkpoints_path,
        'diverse_path': diverse_path,
        'log_file': log_file_path,
        'exp_base':exp_base
    }


def _scientific(float_num):
    # if float_num < 1e-4:
    #    return str(float_num)
    if float_num == 1e-5:
        return '1e-5'
    elif float_num == 5e-5:
        return '5e-5'
    elif float_num == 1e-4:
        return '1e-4'
    elif float_num == 1e-3:
        return '1e-3'
    raise NotImplementedError('In [scientific]: Conversion from float to scientific str needed.')


def extend_path(val_path, number):
    return f'{val_path}_{number}'


def make_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f'In [make_dir_if_not_exists]: created path "{directory}"')


def read_image_ids(data_folder, dataset_name):
    """
    It reads all the image names (id's) in the given data_folder, and returns the image names needed according to the
    given dataset_name.

    :param data_folder: to folder to read the images from. NOTE: This function expects the data_folder to exist in the
    'data' directory.

    :param dataset_name: the name of the dataset (is useful when there are extra unwanted images in data_folder, such as
    reading the segmentations)

    :return: the list of the image names.
    """
    img_ids = []
    if dataset_name == 'cityscapes_segmentation':
        suffix = '_color.png'
    elif dataset_name == 'cityscapes_leftImg8bit':
        suffix = '_leftImg8bit.png'
    else:
        raise NotImplementedError('In [read_image_ids] of Dataset: the wanted dataset is not implemented yet')

    # all the files in all the subdirectories
    for city_name, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(suffix):  # read all the images in the folder with the desired suffix
                img_ids.append(os.path.join(city_name, file))
    return img_ids


def files_with_suffix(directory, suffix):
    files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*{suffix}', recursive=True)]  # full paths
    return files


def get_file_with_name(directory, filename):
    """
    Finds the file with a specific name in a hierarchy of directories.
    :param directory:
    :param filename:
    :return:
    """
    files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/{filename}', recursive=True)]  # full paths
    return files[0]  # only one file should be found with a specific name


def absolute_paths(directory):
    # return [os.path.abspath(filepath) for filepath in os.listdir(directory)]
    return [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*', recursive=True)]  # full paths


def pure_name(path):
    return os.path.split(path)[1]


def city_and_pure_name(filepath):
    return filepath.split(os.path.sep)[-2], pure_name(filepath)
    # splits = filepath.split(os.path.sep)
    # city = splits[-2]
    # name = pure_name(filepath)


def replace_suffix(name, direction):
    if direction == 'segment_to_real':
        return name.replace('_gtFine_color.png', '_leftImg8bit.png')
    return name.replace('_leftImg8bit.png', '_gtFine_color.png')


def get_all_data_folder_images(path, partition, image_type):
    pattern = '_color.png' if image_type == 'segment' else '_leftImg8bit.png'
    files = [os.path.abspath(path) for path in glob.glob(f'{path}/{partition}/**/*{pattern}', recursive=True)]  # full paths
    return files

