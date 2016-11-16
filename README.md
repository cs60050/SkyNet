# SkyNet

Team Name: SkyNet

Area: Computer Generated Art

Term Project, Machine Learning (CS60050), Autumn 2016-2017, IIT Kharagpur

Project Title: "Image Re-Colorization and Extrpolation and cascading of both in different orders to compare the results"

Project Discription: We Are a bunch of Machine Leaning and Computer Vision enthusiasts and plan on comparing results of various approaches to image recolorization through various conventional and non-conventional methodologies

Website: https://cs60050.github.io/SkyNet/website/

# Group Members
[Vineet Jain](https://github.com/VineetJain96)

[Saurabh Dash](https://github.com/saurabhdash)

[Aditya Sinha](https://github.com/adityasinha379)

[Harsh Bajaj](https://github.com/harsh96)

[Preetham KS](https://github.com/preethamks2016)

[Oindrila Saha](https://github.com/oindrilasaha)

[Tejas Nitin Lad](https://github.com/tejasytl)

[Ramit Pahwa](https://github.com/Ramit-Pahwa)

Anukul Jha

Sahil Chadha

Gabriel Werner

### DataSet For Training Purposes

1. http://vision.cs.illinois.edu/projects/lscolor/lscolorization_data.tar.gz


**Idea implemented inspired by:** [http://tinyclouds.org/colorize/](http://tinyclouds.org/colorize/)


# Contents Of This Repository
The folder **models** contains the modified VGG-16 architecture that we have used for image recolorization. Contents: architeture.py

The folder **utils** comprises of 3 scripts:

- *feed_input.py* - to read the images and feed them to the neural network for training/testing

- *convert_images.py* - to convert images from RGB to YUV (for training and grayscale conversion) and vice=versa (for computing RGB colorized output)

- *batchnorm.py* - to apply batch normalisation on the different layers of the VGG-16 neural network

Apart from this, there is a main script *colorize.py* that calls all the functions and performs training/testing.

The script *color_batch.py* calls the trained model and deploys it to colorize a batch of images. 

The folder **examples** has a few test cases on which we ran the code. The first image is the grayscale, the second is the recolorised one and the third is the original image. It also has a 3-minute frame by frame recolorized video of a Charlie Chaplin sequence.

The folder **website** has the code for the [website](https://cs60050.github.io/SkyNet/website/) that we have made for this project.

## Prerequisites
- TensorFlow
- VGG-16 model
- Image Dataset

## Explanation of Architecture
The VGG-16 convolutional network is a standard architecture used for object detection in images using CNN. However, in our PS, we need to not only detect objects in the grayscale image, but also to associate U and V values with it for colorization. For this, we remove the final class labels of the VGG-16 output (truncated VGG-16). Instead, we have used additional architecture, wherein we have first passed the grayscale image through VGG-16. Then, using the highest layer, we have performed batch normalisation to infer some color, which is upscaled and merged with the batch normalised color information from the next highest layer. Proceeding like this, we have worked our way to the bottom of VGG-16 architecture and obtained the UV output. From there, it's only a matter of linear transform to convert orthonormal basis YUV to RGB color output.

### Possibility of improvement
It seems as though for some images, the amount of information retrieved is insufficient to predict coherency. So, we would get better results with the VGG-19 network and would be able to color more of the image.
