import tensorflow as tf
import tensorlayer as tl
import multiprocessing as mp
import matplotlib.pyplot as plt

from easydict import EasyDict as edict
from tensorlayer.layers import Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense
from tensorlayer.models import Model
from tensorflow.keras.preprocessing import image


shuffle_buffer_size = 128


def check_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


def get_config():
    """Get a configuration dict containing hyperparameters and
    paths needed for the training and evaluation of a model.
    """
    config = edict()
    config.TRAIN = edict()

    config.TRAIN.batch_size = 8
    config.TRAIN.lr_init = 1e-4
    config.TRAIN.beta1 = 0.9
    config.TRAIN.n_images = 100

    # config for G initialization
    config.TRAIN.n_epoch_init = 100  # 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 4) # 2

    # adversarial learning (SRGAN)
    config.TRAIN.n_epoch = 100  # 2000
    config.TRAIN.lr_decay = 0.1
    config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)  # 2

    # train set location
    # config.TRAIN.hr_img_path = '/content/drive/MyDrive/AML_final_project/landscapes/TRAIN_HR/'
    config.TRAIN.hr_spec_img_path = '/content/drive/MyDrive/AML_final_project/landscapes/'
    config.TRAIN.hr_img_path = '/content/drive/MyDrive/AML_final_project/DIV2K_train_HR/'
    # config.TRAIN.lr_img_path = '/content/drive/MyDrive/AML_final_project/DIV2K_train_LR_bicubic/X4/'

    config.VALID = edict()
    # test set location
    config.VALID.hr_img_path = '/content/drive/MyDrive/AML_final_project/landscapes/VALID_HR/'
    # config.VALID.hr_img_path = "Final_project/data/landscapes_labels/VALID_HR/"
    # config.VALID.hr_img_path = '/content/drive/MyDrive/AML_final_project/DIV2K_valid_HR/'
    # config.VALID.lr_img_path = '/content/drive/MyDrive/AML_final_project/DIV2K_valid_LR_bicubic/X4/'

    config.save_dir = "/content/drive/MyDrive/AML_final_project/samples"
    config.checkpoint_dir = "/content/drive/MyDrive/AML_final_project/models"
    config.mse_matrix_path = "/content/drive/MyDrive/AML_final_project/Final_matrices/mse_matrix.csv"
    config.ssim_matrix_path = "/content/drive/MyDrive/AML_final_project/Final_matrices/ssim_matrix.csv"

    # config.save_dir = "Final_project/samples"
    # config.checkpoint_dir = "Final_project/models"

    return config


def get_train_data(config, generic=True, land_class=None):
    """Returns a tf.Dataset with images coming from the
    selected folder.

    Set generic to True if you want the div2k dataset,
    otherwise set it to False and assign to land_class an
    integer between 2 and 6 to return the related landscape
    dataset.

    Args:
        config : edict
            Configuration dict of the project.
        generic : boolean
            Return div2k dataset.
        land_class : int or None. 
            Integer in [2-6]. With generic=False, returns the landscape
            dataset of that class.

    Returns:
        A tf.Dataset of images with batch_size and n_images taken from config.
    """
    # load dataset
    if generic:
        path = config.TRAIN.hr_img_path
        regx = '.*.png'
    else:
        path = config.TRAIN.hr_spec_img_path + str(land_class) + "/"
        regx = '.*.jpg'

    train_hr_img_list = sorted(tl.files.load_file_list(
        path=path, regx=regx, printable=False))[:config.TRAIN.n_images]

    train_hr_imgs = tl.vis.read_images(
        train_hr_img_list, path=path, n_threads=32)

    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img

    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.

        hr_patch = tf.image.random_flip_left_right(hr_patch)
        hr_patch = tf.image.random_brightness(hr_patch, max_delta=0.4)
        hr_patch = tf.image.random_contrast(hr_patch, lower=0.1, upper=0.4)
        lr_patch = tf.image.resize(hr_patch, size=[96, 96],
                                   method='bicubic')  # or method='bilinear')
        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(
        generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(
        _map_fn_train, num_parallel_calls=mp.cpu_count())
    # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(config.TRAIN.batch_size)
    # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds


def get_G(input_shape):
    """Get a Generator model with randomly inizialized weights.

    Args:
        input_shape : tuple
            Input shape of the Input layer of the model.
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu,
               padding='SAME', W_init=w_init)(nin)
    temp = n

    def get_G_res_block(n, w_init, g_init):
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME',
                    W_init=w_init, b_init=None)(n)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME',
                    W_init=w_init, b_init=None)(nn)
        nn = BatchNorm2d(gamma_init=g_init)(nn)
        nn = Elementwise(tf.add)([n, nn])
        return nn

    def get_conv_block(n, w_init):
        n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
        n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
        return n

    # B residual blocks
    for _ in range(16):
        n = get_G_res_block(n, w_init, g_init)

    n = Conv2d(64, (3, 3), (1, 1), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=g_init)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    for _ in range(2):
        n = get_conv_block(n, w_init)

    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh,
                padding='SAME', W_init=w_init)(n)
    G = Model(inputs=nin, outputs=nn)  # , name="generator"
    return G


def get_D(input_shape):
    """Get a Discriminator model with randomly inizialized weights.

    Args:
        input_shape : tuple
            Input shape of the Input layer of the model.
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    def lrelu(x): return tl.act.lrelu(x, 0.2)

    nin = Input(input_shape)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu,
               padding='SAME', W_init=w_init)(nin)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME',
               W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME',
               W_init=w_init, b_init=None)(nn)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME',
               W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=gamma_init)(n)
    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])

    n = Flatten()(n)
    no = Dense(n_units=1, W_init=w_init)(n)
    D = Model(inputs=nin, outputs=no)  # , name="discriminator"
    return D


def show_images(im_sr, im_hr):
    titles = ["Original High-res image",
              "Generated SR image"]
    for i, img in enumerate([im_hr, im_sr]):
        plt.figure(figsize=(20, 10))
        plt.title(titles[i])
        im = tf.squeeze(img)
        im = image.array_to_img(im)
        plt.imshow(im)
        plt.show()


def cropping(img):
    hr_patch = tf.image.random_crop(img, [384, 384, 3])
    hr_patch = (tf.cast(hr_patch, tf.float32) / 127.5) - \
        1.  # or method='bilinear')
    lr_patch = tf.image.resize(hr_patch, size=[96, 96], method='bicubic')
    return lr_patch, hr_patch
