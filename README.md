# SkyNet

Team Name: SkyNet

Area: Computer Generated Art

Term Project, Machine Learning (CS60050), Autumn 2016-2017, IIT Kharagpur

Project Title: "Image Re-Colorization and Extrpolation and cascading of both in different orders to compare the results"

Project Discription: We Are a bunch of Machine Leaning and Computer Vision enthusiasts and plan on comparing results of various approaches to image recolorization through various conventional and non-conventional methodologies

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

# DataSet For Training Purposes

1. http://vision.cs.illinois.edu/projects/lscolor/lscolorization_data.tar.gz

**Idea implemented inspired by:** [http://tinyclouds.org/colorize/](http://tinyclouds.org/colorize/)

# Contents Of This Repository
The folder **models** contains the modified VGG-16 architecture that we have used for image recolorization. Contents: architeture.py
The folder **utils** comprises of 3 scripts:
-feed_input.py - to read the images and feed them to the neural network for training/testing
-convert_images.py - to convert images from RGB to YUV (for training and grayscale conversion) and vice=versa (for computing RGB colorized output)
-batchnorm.py - to apply batch normalisation on the different layers of the VGG-16 neural network
