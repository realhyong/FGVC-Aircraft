import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers
import tensorflow.keras as keras


class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_out= layers.GlobalAveragePooling2D()         # 空间的全局平均池化，shape(1*1*C)
        self.max_out= layers.GlobalMaxPooling2D()             # 空间的最大池化，shape(1*1*C)

        self.fc1 = layers.Dense(in_planes//ratio, activation='relu', use_bias=True, bias_initializer='zeros')       # 全连接层1，神经元个数为： 通道数C/压缩比R
        self.fc2 = layers.Dense(in_planes, use_bias=True, bias_initializer='zeros')                                 # 全连接层2，神经元个数为：通道数C

    def call(self, inputs):
        avg_out = self.avg_out(inputs)                # 经过全局平均池化操作, shape=(None, feature)              
        max_out = self.max_out(inputs)                # 经过最大池化操作, shape=(None, feature)
        
        avg_out = layers.Reshape((1, 1, avg_out.shape[1]))(avg_out)   # shape (None, 1, 1, feature)
        max_out = layers.Reshape((1, 1, max_out.shape[1]))(max_out)   # shape (None, 1, 1, feature)

        avg_out = self.fc2(self.fc1(avg_out))          # shape (None, 1, 1, feature)
        max_out = self.fc2(self.fc1(max_out))          # shape (None, 1, 1, feature)

        x = avg_out + max_out
        x = tf.nn.sigmoid(x)                 
          
        return x


############################### 空间注意力机制 ###############################
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()       
        self.conv1 = layers.Conv2D(1, kernel_size=kernel_size, strides=1, activation='sigmoid',padding='same', use_bias=False)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        x = tf.stack([avg_out, max_out], axis=-1)             # 创建一个维度,拼接到一起concat。
        x = self.conv1(x)

        return x


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        ############################### 注意力机制 ###############################
        self.ca = ChannelAttention(out_channel * self.expansion)
        self.sa = SpatialAttention()

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        ############################### 注意力机制 ###############################
        x = self.ca(x) * x
        x = self.sa(x) * x
       
        x = self.add([x, identity])
        x = self.relu(x)

        return x



def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


