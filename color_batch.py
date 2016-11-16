import tensorflow as tf
import skimage.transform
from skimage.io import imsave, imread

import os, sys

# Open a file
path_o = "imgs/"
dirs = os.listdir( path_o )


def load_image(path):
    img = imread(path)
    print(img.shape)
    # resize
    img = skimage.transform.resize(img, (224, 224))
    
    return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0

for file in dirs:
	folder = "imgs_f/"
        fullpath= path_o+file
        print fullpath
	
	gray = load_image(fullpath).reshape(1, 224, 224, 1)

	with open("colorize.tfmodel", mode='rb') as f:
	    fileContent = f.read()
        with tf.Graph().as_default():
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fileContent)
		grayscale = tf.placeholder("float", [1, 224, 224, 1])
		inferred_rgb, = tf.import_graph_def(graph_def, input_map={"grayscale": grayscale },
		                            return_elements=["inferred_rgb:0"])

		fullsavepath=folder+file

		with tf.Session() as sess:
		    inferred_batch = sess.run(inferred_rgb, feed_dict={ grayscale: gray })
		    imsave(fullsavepath, inferred_batch[0])
		    

