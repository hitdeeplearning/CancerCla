# 参考代码 https://keras.io/examples/cifar10_resnet/#trains-a-resnet-on-the-cifar10-dataset
import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
from classifiers.flat.resnet.liubo_2D_dataset import liubo_2D_dataset_c
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy, binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import time
import shutil


# resnet v1 配置
model_name = "resnet_v1_2D_classifier"
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
configs = {
    "train_dir_list":["/home/liubo/data/graduate/liubo_2D_dataset/origin_5_fold/fold1",
                      "/home/liubo/data/graduate/liubo_2D_dataset/origin_5_fold/fold2",
                      "/home/liubo/data/graduate/liubo_2D_dataset/origin_5_fold/fold3",
                      "/home/liubo/data/graduate/liubo_2D_dataset/origin_5_fold/fold4"],
    "val_dir": "/home/liubo/data/graduate/liubo_2D_dataset/origin_5_fold/fold0", # 验证集不用增强
    "test_dir" :"/home/liubo/data/graduate/liubo_2D_dataset/origin_5_fold/fold0",  # 测试在 predict中用到
    "batch_size" : 8,
    "log_dir":"./logs/"+ model_name+time_str,
    "work_dir":"/home/liubo/nn_project/LungSystem/workdir/" + model_name+time_str,
    "model_name": model_name,
    "model_save_path":"/home/liubo/nn_project/LungSystem/models/guaduate/" + model_name,
    "learn_rate":0.00001,
    "epoches":200,
    "version":1
}


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal', # 权值初始化 https://blog.csdn.net/qq_27825451/article/details/88707423
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=5):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):

            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])   # 注意这里用add 而不是conatenated
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)   # 这里的Dense 就是全联接

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=5):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_net(load_weight_path=None):
    version = configs["version"]
    n = 3
    # 计算depth
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # input_shape
    input_shape = (512,512,1)  # 512*512 的灰度图
    
    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(optimizer=SGD(lr=configs["learn_rate"], 
                                momentum=0.9, 
                                nesterov=True),
                  loss=categorical_crossentropy, 
                  metrics=[categorical_crossentropy, categorical_accuracy])

    model.summary()
    model_type = 'ResNet%dv%d' % (depth, version)
    print(model_type)
    return model


def train(model_name,load_weight_path=None):
    train_dir_list = configs["train_dir_list"]
    val_dir = configs["val_dir"]
    test_dir = configs["test_dir"]
    batch_size = configs["batch_size"]
    work_dir = configs["work_dir"]
    model_name = configs["model_name"]
    

    dataset = liubo_2D_dataset_c(train_dir_list,val_dir,test_dir,batch_size)
    dataset.prepare_train_val_dataset()
    train_dataset, val_dataset = dataset.get_train_val_dataset()
    x_train, y_train = zip(*train_dataset)
    x_val,y_val = zip(*val_dataset)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    model = get_net()

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    checkpoint = ModelCheckpoint(filepath=work_dir+ "/" + model_name + "_" + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=False, 
                                 save_weights_only=False, 
                                 mode='auto', 
                                 period=1)
    # 每隔一轮且每当val_loss降低时保存一次模型
    best_model_path = work_dir + "/" + model_name + "_best.hd5"
    checkpoint_fixed_name = ModelCheckpoint(filepath = best_model_path,
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            save_weights_only=False, 
                                            mode='auto', 
                                            period=1)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=configs["epoches"],
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=[checkpoint, 
                         checkpoint_fixed_name, 
                         TensorBoard(log_dir=configs["log_dir"],
                                     write_images=True)])

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_save_path = configs["model_save_path"]
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_name = configs["model_name"]
    work_dir = configs["work_dir"]
    train(model_name=model_name)
    best_model_path = work_dir + "/" + model_name + "_best.hd5"
    shutil.copy(best_model_path, model_save_path)


# # Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])