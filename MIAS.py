from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import re
from six.moves import xrange  # pylint: disable=redefined-builtin
import pandas as pd
import utils
import scipy

# Function for obtaining center crops from an image
def crop_center(x, crop_w, crop_h):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return x[j:j + crop_h, i:i + crop_w]


# More general function for cropping a patch around a point
# x,y define the upper left corner for the cropping
# Filtering black only patches in this function.
# Also transfrom the batches in the function to pixel values between -1,1
def crop_around(img, x, y, width, height):
    h, w = img.shape[:2]
    new_image = img[y:y + height, x:x + width]
    sumnone=0
    if (np.sum(new_image) < 0.19 * x * y):
        sumnone+=1
        print("none",sumnone)
        return None
    return utils.transform(new_image, height, width, height, width)


# Function takes an array of images and takes k random crops from each image
# Sampling is currently with replacement
# Again expecting a numpy array
def rand_crop(images, crop_w, crop_h, k):
    num_images = images.shape[0]
    h, w = images.shape[1:3]
    print(h, w)
    # define the return array
    crops = np.zeros([num_images * k, crop_h, crop_w])
    # generate random points to crop around
    x = np.random.randint(0, w - crop_w, [num_images, k])
    y = np.random.randint(0, h - crop_h, [num_images, k])
    # crop
    for idx, im in enumerate(images):
        for counter in xrange(k):
            crops[(idx * k) + counter] = crop_around(im, x[idx, counter], y[idx, counter], crop_w, crop_h)
    return crops


# This function takes sequential crops of the image with defined step size
# the skip parameter defines a  probability of skiping a path.
# This might be better than rand_crop because it samples without replacement, and we can control the amount of overlap between crops
def seq_crop(images, crop_w, crop_h, step=10, skip=0.1):
    num_images = images.shape[0]
    h, w = images.shape[1:3]
    crops = []
    for idx, im in enumerate(images):
        tmp = []
        for x in xrange(0, w - crop_w, step):
            for y in xrange(0, h - crop_h, step):
                if (np.random.random() > skip):
                    c = crop_around(im, x, y, crop_w, crop_h)
                    if (c is not None):
                        crops.append(c)
                    # following two lines can be used to output an array where the crops from each image are in a separate list
                    # tmp.append(c)
                    # crops.append(np.asarray(tmp))
    print("in seq crop", np.asarray(crops).shape)
    print(images.shape)
    return np.asarray(crops)


# Function for reading PGM files
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

def readFaces():
    images=[]
    for i in range(1,5000):
        x="{0:0=6d}".format(i)
        print(x)
        images.append(imread('data/img_align_celeba/img_align_celeba/'+str(x)+'.jpg'))
    print("tale3 men 3andi ", np.asarray(images).shape)
    return images

def readMiasDATA():
    # read csvFile
    data_train = pd.read_csv('mias.csv')
    abnormal = data_train[data_train.abnormality_class != 'NORM']
    normal = data_train[data_train.abnormality_class == 'NORM']
    data_train
    # HELPER FUNCTIONS
    data_train[data_train.abnormality_class == 'NORM']["abnormality_class"]
    images = []
    for i, row in normal.iterrows():
        images.append(read_pgm('DataSet/' + row['reference_number'] + '.pgm'))
    # Removing noise-medical labels-from data, by overlaying corners with black pixels
    j = 0;
    for i, row in normal.iterrows():
        images[j].setflags(write=1)
        if (int(row['reference_number'][-3:]) % 2 == 0):
            images[j][:324, 700:1024] = np.zeros((324, 324))
        else:
            images[j][:324, :324] = np.zeros((324, 324))
        # matplotlib.image.imsave('DataSet/' + row['reference_number'] + '.png', images[j], vmin=0, vmax=255, cmap='gray')
        j += 1
    return images, [t for t in data_train.abnormality_class if t == 'NORM']

    ################################################################TEMP
    # read csvFile
    data_train = pd.read_csv('mias.csv')
    abnormal = data_train[data_train.abnormality_class != 'NORM']
    normal = data_train[data_train.abnormality_class == 'NORM']
    data_train
    # HELPER FUNCTIONS
    data_train[data_train.abnormality_class == 'NORM']["abnormality_class"]
    images = []
    for i, row in normal.iterrows():
        images.append(read_pgm('DataSet/' + row['reference_number'] + '.pgm'))
    return images, data_train[data_train.abnormality_class == 'NORM']["abnormality_class"]


def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)



plt.ion()

from skimage.transform import rotate
from skimage.exposure import rescale_intensity
from random import randint

from skimage.io import imshow, show, imsave
import skimage


# DATASET CLASS
class MIAS(object):
    # class Options(object):


    def __init__(self, images, labels=None, sets=[1, 0, 0], cache=False, classes_of_interest=[], options=dict()):
        # Attributes:
        # Dictionary    self._classes
        # String        self._path
        # np.array      self._images
        # np.array      self._labels
        # np.array      self._set (train = 0, val = 1, test = 2)
        # int           w
        # MIAS.Options  self._options
        # Sets is a 3 tuple describing [training:validation:test] partition
        # All arrays are numpy inside the class. AYAD

        self._images = np.asarray(images)
        self._sets = np.asarray(sets)
        self._classes_of_interest = np.asarray(classes_of_interest)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._batch_count = 0
        if (labels is not None):
            self._labels = np.asarray(labels)
        else:
            self._labels = labels

        # Options is a dict containing the cropping and augmentation options.
        # cropping specifies croping mode: full,center,random_crops,seq_crop. We might not implement all those
        # k_crop specifies number of crops to take in the case of random crops
        # crop_size specifies the size of the crops [2] (width , height)
        # crop_step a parameter for seq_crop
        # crop_skip a parameter for seq_crop
        # rotation specifies the rotation params for augmentation [2]
        # noise specifies the amount of noise to be added during augmentation
        # intensity_scaling specifies the intensity scaling during augmentation[2]
        self._options = options
        print("entering augment")
        # self.augmentDataSet(self._images, [1,10], 0.04, [(0,200),(10,255)])
        print("entering crop")
        self.crop()
        print("init done")

        # self.stats()

    # if cache == True, then try to load the object from disk using loadFromDisk
    # if there is no object on disk, just create the all the data and then call storeToDisk()

    # According to options, augment the dataset
    # use either full images, or k random crops, or center crops around labels
    # get only images/crops from specific class(es) or all classes
    # rotation, zoom in/out, add noise, intensity scaling
    # specify width, height of patches / images

    @property
    def classes(self):
        return self._classes

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def sets(self):
        return self._sets

    @property
    def num_data(self):
        return self._images.shape[0]

    @property
    def num_channels(self):
        if (len(self._images.shape) <= 3):
            return 1
        else:
            return self._images.shape[3]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def label_dim(self):
        if (self._labels is not None):
            return self._labels.shape[0]
        else:
            return False

            # augmentDataSet(images,[1,10],0.0004,[(0,200),(10,255)])

    def augmentDataSet(self, images, rotation, noise, intensityScale):
        j = 0;
        temp = []
        for im in images:
            print(j)
            rot_pic = rotate(im, randint(rotation[0], rotation[1]))
            # imsave('DataSetAug/'+ str(j) +'_rot.png',rot_pic,vmin=0,vmax=255,cmap='gray')
            temp.append(rot_pic)
            noise_pic = skimage.util.random_noise(im, mode='gaussian', var=noise)
            # imsave('DataSetAug/'+ str(j) +'_noise.png',noise_pic,vmin=0,vmax=255,cmap='gray')
            temp.append(noise_pic)
            scale_pic = rescale_intensity(im, in_range=intensityScale[0])
            # imsave('DataSetAug/'+ str(j) +'_scale1.png',scale_pic,vmin=0,vmax=255,cmap='gray')
            temp.append(scale_pic)
            scale_pic = rescale_intensity(im, in_range=intensityScale[1])
            # imsave('DataSetAug/'+ str(j) +'_scale2.png',scale_pic,vmin=0,vmax=255,cmap='gray')
            temp.append(scale_pic)
            j += 1
        np.concatenate((self._images, np.asarray(temp)), axis=0)

    def getName(self):
        string = (str(self._options))
        return ''.join(e for e in string if e.isalnum())

    # Store the entire model and all its data to disk, such that the computation
    # of augmented samples does not have to be repeated everytime
    def storeToDisk(self):
        # TODO
        # file_pi = open(self.getName'.obj', 'w')
        print('b')

    # Use getName() to find out the name of the object you want to load from disk
    def loadFromDisk(self):
        # TODO
        print('b')

    # Visualize the images together with their labels using matplotlib & pyplot
    def visualize(self):
        x = 'a'
        while (x != 'q'):
            rand = np.random.randint(0, self.num_data)
            plt.figure(self._labels[rand])
            plt.imshow(self._images[rand], cmap='gray')
            plt.show(block=True)
            x = input("please press q to quit visualization, any other key to continue\n")


        # crops the data according to options

        # optionally does a shuffling of the data after croping
        # currently implementing full,center, and random crops

    def crop(self, shuffle=True):
        mode = self._options["cropping"]
        w, h = self._options["crop_size"]

        tmp=[]
        for idx, im in enumerate(self._images):
            tmp.append( crop_center(im, 176, 176) ) # Current hack to make things easy
        del self._images
        self._images= np.asarray(tmp)
        if (mode == "center"):
            for im in self._images:
                im = crop_center(im, w, h)
        elif (mode == "seq"):
            step = self._options['crop_step']
            skip = self._options['crop_skip']
            self._images = seq_crop(self._images, w, h, step, skip)
        elif (mode == "rand"):
            k = self._options['k_crop']
            self._images = rand_crop(self._images, w, h, k)
        if (shuffle):
            np.random.shuffle(self._images)

            # Print out statistics of the current dataset, such as number of images, classes, class distribution, set distribution

    def stats(self):
        print("Number of images = ", self._images.shape[0])
        print("Image size = ", self._images.shape[1], " * ", self._images.shape[2])
        # print number of lables
        print("Number of classes = ", self._classes_of_interest.shape[0])
        # print class distribution
        class_count = dict()
        for i in self._classes_of_interest:
            class_count[i] = 0
        for l in self._labels:
            class_count[l] += 1
        print("class distribution : ", class_count)
        print("Set distribution[Train Validation Test]  = ", self.sets / sum(self.sets) * 100)
        # calculate pixel mean and std deviation and display them as images
        m = np.mean(self._images, axis=0)
        std = np.std(self._images, axis=0)
        plt.figure("Mean pixel value")
        plt.imshow(m, cmap='gray')
        plt.figure("Pixel std")
        plt.imshow(std, cmap='gray')
        plt.show(block=True)


        # Very important!!! This method should be called to feed your GAN with training data
        # return a random subset of _images, _labels, _set

    def next_batch(self, batch_size, shuffle=True, setRestriction=None):
        """Return the next `batch_size` examples from this data set."""
        # Right now the assumption is that _images contains the agmented data to be used
        # https://arxiv.org/pdf/1202.4184v1.pdf indicates that sampling without replacement is better
        # Our DCGAN was also working without replacement.
        # So currently, this function works without replacement
        # and if shuffle is desired, data is shuffled in the begining of each epoch

        if (shuffle):
            if (self._batch_count == 0):  # we are at the begining of an epoch
                np.random.shuffle(self._images)

        x = self._images[self._batch_count: self._batch_count + batch_size]
        if (self._labels is not None):
            y = self._labels[self._batch_count: self._batch_count + batch_size]
        self._batch_count += batch_size
        if (self._batch_count > self.num_data):
            self._epochs_completed += 1
            diff = self._batch_count - self.num_data
            self._batch_count = diff
            tmp = self._images[0: diff]
            print(diff, tmp.shape)
            x = np.concatenate((x, tmp))
            if (self._labels is not None):
                tmp2 = self._labels[0: diff]
                y = np.concatenate((y, tmp2))
        elif (self._batch_count == self.num_data):
            self._epochs_completed += 1
            self._batch_count = 0

        if (self.num_channels == 1):  # to be compatible with DCGAN code
            x = np.expand_dims(x, axis=3)

        if (self._labels is not None):
            return x, y
        else:
            return x, None

# data, labels = readMiasDATA()
# options = {"cropping": "seq", "crop_step": 40, "crop_skip": 0.3, "crop_size": (200, 200)}
# mias = MIAS(data[0:5], labels[0:5], [1, 0, 0], False, ["NORM"], options)
#
# print(mias.num_channels)
# In[46]:

#
# options= {"cropping" : "seq", "crop_step" : 200, "crop_skip" : 0.4, "crop_size":(200,200)}
# x = MIAS(images[:30], data_train[data_train.abnormality_class == 'NORM']["abnormality_class"], [80,20,0],False, ["NORM"], options  )
# im,y=x.next_batch(100)
# print(im.shape,"***********")
# plt.imshow(im[0] , cmap='gray')
# plt.show(block=True)
#
# for counter, i in enumerate(im):
#     matplotlib.image.imsave('DataSet_crops/'+ str(counter)+'7a7a.png',i,vmin=0,vmax=255,cmap='gray')
#
