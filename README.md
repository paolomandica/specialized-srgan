# AML Final Project - Specialized SR-GAN

Alessio Sampieri  
Lorenzo Ceccomancini  
Michele Meo  
Paolo Mandica

This is the repository for the final project of the Advanded Machine Learning course for the Master's Degree in Data Science @ Sapienza University of Rome.

The project is described in detail in the report.

The notebook of the project can be found [here](https://colab.research.google.com/drive/1zu3-bMXEdxpidi2KKgA8f87QoRD9tUmB?usp=sharing) (Google Colab).

## Objective

The project is focused on the development of a Specialized Super-Resolution Generative Adversarial Network.

The super-resolution image restoration problem is central in the computer vision field and it can have a lot of applications or utilities. Convolutional neural networks are widely used in this computer vision area and, recently, the SR-GAN architecture has been proposed, in the Generative Adversarial Network for image Super Resolution [paper by Ledig et al.](https://arxiv.org/pdf/1609.04802.pdf), that is able to obtain the super resolution version of a photo-realistic image downscaled by 4x factor.  
In this work we tried to understand this architecture and use it to reconstruct the high-definition version of a specific class of images, in our case landscape images: in particular, through the transfer learning technique, we tried to specialize the pre-trained model, proposed in this [github repository](https://github.com/tensorlayer/srgan) from the official paper, in order to be able to generate high-resolution photos of a specific landscape class.

Our specialized-SR-GAN is able to obtain better results, so lower loss discrepancies from the original high resolution set of images, on almost all the landscape classes we have worked on and we will show this in a matrix specialized models-classes comparison.

![](https://paperswithcode.com/media/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png)


## Datasets
- **DIV2K dataset**: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- **Landscapes dataset**: https://www.kaggle.com/arnaud58/landscape-pictures
