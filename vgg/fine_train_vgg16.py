from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import vgg
import tensorflow as tf
import json
import os
import glob


def main():
    model = vgg("vgg16", 224, 224, 100)
    model.summary()
if __name__ == '__main__':
    main()
