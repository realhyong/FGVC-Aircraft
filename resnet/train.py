import os
# import sys
import glob
import json
# import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import regularizers
# from keras import backend as K 
from matplotlib import pyplot as plt
# from tqdm import tqdm
from model import resnet50
# from cbammodel import resnet50
# from sklearn.model_selection import StratifiedKFold

def main():
    work_root = os.path.abspath('.')  # get data root path
    image_path = os.path.join(work_root, "images")  # data set path
    train_dir = os.path.join(image_path, "trainval")
    # validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    # assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    num_classes = 100

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    def pre_function(img):
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(validation_split=0.1,
                                               horizontal_flip=True,
                                               preprocessing_function=pre_function)


    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical',
                                                               subset='training')

    total_train = train_data_gen.n


    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical',
                                                                  subset='validation')
    
    # img, _ = next(train_data_gen)
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    # K.set_learning_phase(0)
    feature = resnet50(num_classes=100, include_top=False)

    pre_weights_path = os.path.join(work_root, 'resnet', 'tf_resnet50_weights', 'pretrain_weights.ckpt')
    assert len(glob.glob(pre_weights_path+"*")), "cannot find {}".format(pre_weights_path)
    feature.load_weights(pre_weights_path)
    feature.trainable = False
    feature.summary()

    # K.set_learning_phase(1)
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                #  tf.keras.layers.Dropout(rate=0.5),
                                tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(5e-4)),
                                #  tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()
                                 ])
    # model.build((None, 224, 224, 3))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
    #                                               patience=5, 
    #                                               verbose=1,
    #                                               mode = 'min'),
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                      factor = 0.1,
                                                      patience=5, 
                                                      verbose=1,
                                                      min_lr=0.000005),
    # callbacks = [          
                tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/resNet_50.ckpt',
                                                            save_best_only=True,
                                                            save_weights_only=True,
                                                            monitor='val_loss')]

    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        # validation_split=0.1,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

        
    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()