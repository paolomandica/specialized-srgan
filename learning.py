import tensorlayer as tl
import tensorflow as tf
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *


def train(g_pretrained=False, n_trainable=None,
          generic=True, land_class=None, config=None):
    """Train a SRGAN. Choose between training from zero or using a pretrained
    model. You can also choose between training a generic SRGAN or a specialized
    one.

    The model (h5 file) will be saved in the folder indicated in the config edict.

    Args:
        g_pretrained : boolean
            True to use a pretrained mode, False to train from random weights.
        n_trainable : integer
            Number of layer to use for transfer learning. Only if g_pretrained
            is True.
        generic : boolean
            Whether to train a generic model or a specialized one.
        land_class : int or None
            If generic=False, which landscapes class to train the specialized
            model on.
        config : edict of None
            None to use a standard config dict. Input a new config dict to use
            a personalized one.
    """

    if generic == False and land_class == None:
        raise ValueError(
            "If you are training a Specialized-SRGAN, you have to select a landscape class among [2,...,7].")

    if generic and land_class is not None:
        print("WARNING: if you set generic=True, the land_class variable will not be used.")

    total_time = time.time()
    if config == None:
        config = get_config()

    ### HYPER-PARAMETERS ###
    batch_size = config.TRAIN.batch_size
    lr_init = config.TRAIN.lr_init
    beta1 = config.TRAIN.beta1
    # initialize G
    n_epoch_init = config.TRAIN.n_epoch_init
    # adversarial learning (SRGAN)
    n_epoch = config.TRAIN.n_epoch
    lr_decay = config.TRAIN.lr_decay
    decay_every = config.TRAIN.decay_every
    n_images = config.TRAIN.n_images

    # create folders to save result images and trained models
    save_dir = config.save_dir
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = config.checkpoint_dir
    tl.files.exists_or_mkdir(checkpoint_dir)

    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tl.models.vgg16(pretrained=True, end_with='pool4', mode='static')

    if generic:
        g_name = 'g'
        d_name = 'd'
    else:
        g_name = 'g_spec_' + str(land_class)
        d_name = 'd_spec_' + str(land_class)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data(config, generic, land_class)

    trainable_weights = G.trainable_weights

    if g_pretrained and n_trainable:
        nt = -n_trainable-1
        trainable_weights = G.all_weights[:nt]
        G.load_weights(os.path.join(checkpoint_dir, 'g_srgan.npz'))
    else:
        nt = len(G.all_weights)

    g_init_losses = []
    # initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in train_ds.enumerate():
            # if the remaining data in this epoch < batch_size
            if lr_patchs.shape[0] != batch_size:
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                G.all_weights[:nt] = trainable_weights
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(
                    fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(
                save_dir, 'train_g_init_{}.png'.format(epoch)))
    # aggiunto Ale
    tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(
        save_dir, 'train_g_init_final.png'))

    # adversarial learning (G, D)
    n_step_epoch = round(n_images // batch_size)  # era n_epoch //
    g_losses = []
    d_losses = []
    for epoch in range(n_epoch):
        g_losses_epoch = []
        d_losses_epoch = []
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            # if the remaining data in this epoch < batch_size
            if lr_patchs.shape[0] != batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                # the pre-trained VGG uses the input range of [0, 1]
                feature_fake = VGG((fake_patchs+1)/2.)
                feature_real = VGG((hr_patchs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(
                    logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(
                    logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2  # discriminator loss
                g_gan_loss = 1e-3 * \
                    tl.cost.sigmoid_cross_entropy(
                        logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(
                    fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * \
                    tl.cost.mean_squared_error(
                        feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss  # generator loss
            grad = tape.gradient(g_loss, trainable_weights)
            g_optimizer.apply_gradients(zip(grad, trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))
            g_losses_epoch.append(g_loss.numpy())
            d_losses_epoch.append(d_loss.numpy())
            G.all_weights[:nt] = trainable_weights
        g_losses.append(g_losses_epoch)
        d_losses.append(d_losses_epoch)
        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (
                lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 5 == 0):  # era epoch%10
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(
                save_dir, 'train_{}_{}.png'.format(g_name, epoch)))
            G.save_weights(os.path.join(
                checkpoint_dir, '{}.h5'.format(g_name)))
            D.save_weights(os.path.join(
                checkpoint_dir, '{}.h5'.format(d_name)))

    tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(
        save_dir, 'train_{}_final.png'.format(g_name)))
    G.save_weights(os.path.join(checkpoint_dir, '{}.h5'.format(g_name)))
    D.save_weights(os.path.join(checkpoint_dir, '{}.h5'.format(d_name)))

    if not g_pretrained:
        pd.DataFrame(g_init_losses).to_csv(os.path.join(
            checkpoint_dir, g_name + '_init_loss.csv'))
    pd.DataFrame(g_losses).to_csv(os.path.join(
        checkpoint_dir, g_name + '_losses.csv'))
    pd.DataFrame(d_losses).to_csv(os.path.join(
        checkpoint_dir, d_name + '_losses.csv'))
    print('TOTAL_TIME: ', round((time.time()-total_time)/60, 2), 'min')


def evaluate(imid=None, landscapes=False, generic=True,
             land_class=None, config=None):
    """Evaluate a SRGAN. Choose between evaluating a generic SRGAN or a specialized
    one and on which class and image to do the evaluation.

    Use the funtion utils.show_images to display the high-resolution and the
    super-resolution images returned.

    Args:
        imid : int or None
            Id of the image to generate in super-resolution.
        generic : boolean
            Whether to evaluate a generic model or a specialized one.
        land_class : int or None
            If generic=False, which specialized model to use.
        config : edict or None
            None to use a standard config dict. Input a new config dict to use
            a personalized one.

    Returns:
        Generated image, high-res image, mse loss, 1-ssim loss.
    """

    if generic == False and land_class == None:
        raise ValueError(
            "If you are evaluating a Specialized-SRGAN, you have to select a landscape class among [2,...,7].")

    if generic and land_class is not None:
        print("WARNING: if you set generic=True, the land_class variable will not be used.")

    if config == None:
        config = get_config()

    if imid == None:
        imid = 30

    if generic:
        g_name = 'g'
    else:
        g_name = 'g_spec_' + str(land_class)

    ###====================== PRE-LOAD DATA ===========================###
    if landscapes:
        valid_hr_img_list = sorted(tl.files.load_file_list(
            path=config.VALID.hr_img_path, regx='.*.jpg', printable=False))
        valid_hr_imgs = tl.vis.read_images(
            valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
        valid_hr_img = valid_hr_imgs[imid]
        w, h, c = valid_hr_img.shape
        valid_lr_img = tf.image.resize(
            valid_hr_img, size=[w//4, h//4], method='bicubic')
    else:
        valid_lr_img_list = sorted(tl.files.load_file_list(
            path=config.VALID.lr_img_path, regx='.*.png', printable=False))
        valid_lr_imgs = tl.vis.read_images(
            valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = None

        ###========================== DEFINE MODEL ============================###

    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    G = get_G([1, None, None, 3])
    # 'g.h5' 'g_spec.h5 'g_srgan.npz'
    G.load_weights(os.path.join(config.checkpoint_dir, '{}.h5'.format(g_name)))
    G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis, :, :, :]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]
    valid_lr_crop, valid_hr_crop = cropping(valid_hr_img)
    fake_patchs = G(tf.expand_dims(valid_lr_crop, 0))
    mse_loss = tl.cost.mean_squared_error(
        tf.squeeze(fake_patchs), valid_hr_crop, is_mean=True)
    ssim_loss = tf.image.ssim(tf.squeeze(fake_patchs), valid_hr_crop, 126.5)
    out = G(valid_lr_img).numpy()

    # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("LR size: %s /  generated HR size: %s" % (size, out.shape))
    print("[*] save images")
    image_name = "valid_" + g_name + ".png"
    tl.vis.save_image(out[0], os.path.join(config.save_dir, image_name))
    tl.vis.save_image(valid_lr_img[0], os.path.join(
        config.save_dir, 'valid_lr.png'))
    # if valid_hr_img is not None:
    # tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = tf.image.resize(valid_lr_img[0], size=[
                               size[0] * 4, size[1] * 4], method='bicubic')
    tl.vis.save_image(out_bicu, os.path.join(
        config.save_dir, 'valid_bicubic.png'))

    return out, valid_hr_img, mse_loss, (1-ssim_loss)


def plot_loss(config=None, generic=True, land_class=None):
    """Plot the training loss history of Generator and Discriminator.

    Args:
        config : edict or None
            None to use a standard config dict. Input a new config dict to use
            a personalized one.
        generic : bool
            Whether to plot loss history of a generic model or not.
        land_class: int or None
            If generic=False, plot the loss history of the selected specialized model.
    """

    if config == None:
        config = get_config()

    if generic:
        g_name = 'g_losses.csv'
        d_name = 'd_losses.csv'
    else:
        g_name = 'g_spec_{}_losses.csv'.format(str(land_class))
        d_name = 'd_spec_{}_losses.csv'.format(str(land_class))

    g_losses = pd.read_csv(os.path.join(
        config.checkpoint_dir, g_name))
    l = g_losses.iloc[:, 1:].mean(axis=1)
    plt.plot(range(len(l)), l, )
    plt.xlabel('Epoch')
    plt.ylabel('Content Loss')
    plt.title('Generator (GAN)')
    plt.text(3, np.max(l)-0.001, 'last='+str(round(l.iloc[-1], 5)))
    plt.show()

    d_losses = pd.read_csv(os.path.join(
        config.checkpoint_dir, d_name))
    l = d_losses.iloc[:, 1:].mean(axis=1)
    plt.plot(range(len(l)), l, )
    plt.xlabel('Epoch')
    plt.ylabel('Adversarial Loss')
    plt.title('Discriminator (GAN)')
    plt.text(3, np.max(l)-0.01, 'last='+str(round(l.iloc[-1], 5)))
    plt.show()


def compute_matrices(config=None):
    """Compute and save the matrices of mse and ssim losses.
    """
    all_pre = [2, 3, 4, 5, 6, None]
    mse_matrix = []
    ssim_matrix = []

    n_imm = 20

    if config == None:
        config = get_config()

    for mod in all_pre:
        mse_class = []
        ssim_class = []
        for cls in all_pre:
            if cls == None:
                config.VALID.hr_img_path = config.TRAIN.hr_spec_img_path + \
                    "/" + str(1) + '/'
            else:
                config.VALID.hr_img_path = config.TRAIN.hr_spec_img_path + \
                    "/" + str(cls) + '/'
            mse_array = []
            ssim_array = []
            for img in range(n_imm):
                print('################ model: ', mod, ' ### cls: ',
                      cls, ' ### img: ', img, ' ############')
                if mod == None:
                    im_sr, im_hr, mse, ssim = evaluate(-img,
                                                       landscapes=True, generic=True, land_class=mod)
                else:
                    im_sr, im_hr, mse, ssim = evaluate(-img,
                                                       landscapes=True, generic=False, land_class=mod)
                mse_array.append(mse.numpy())
                ssim_array.append(ssim.numpy())
            mse_class.append(np.mean(mse_array))
            ssim_class.append(np.mean(ssim_array))
        mse_matrix.append(mse_class)
        ssim_matrix.append(ssim_class)

    pd.DataFrame(mse_matrix).to_csv(config.mse_matrix_path)
    pd.DataFrame(ssim_matrix).to_csv(config.ssim_matrix_path)
