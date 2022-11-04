import matplotlib.pyplot as plt
import tensorflow as tf

from learning import train, evaluate
from utils import show_images, get_config


if __name__ == "__main__":

    # reduce_gpu()
    config = get_config()

    # training
    train(g_pretrained=True, n_trainable=1, generic=False, config=config)

    # # evaluation
    # im_sr, im_hr, _, _ = evaluate(
    #     15, landscapes=True, generic=True, land_class=None)

    # show_images(im_sr, im_hr)
