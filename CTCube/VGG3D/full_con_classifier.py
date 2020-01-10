import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
from classifiers.graduate.resample_dataset import ClassificationDataset
from keras.models import Model 
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten, Dropout, AveragePooling3D, Dense
from keras.metrics import categorical_accuracy, binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
import time
import shutil

# base 配置
# model_name = "full_con_classifier_base"
# # time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# time_str = "20200104_231603"
# configs = {
#     "train_dir_list":["/home/liubo/data/graduate/resampled_classification_dataset/resample/fold1",
#                       "/home/liubo/data/graduate/resampled_classification_dataset/resample/fold2",
#                       "/home/liubo/data/graduate/resampled_classification_dataset/resample/fold3",
#                       "/home/liubo/data/graduate/resampled_classification_dataset/resample/fold4"],
#     "val_dir": "/home/liubo/data/graduate/resampled_classification_dataset/origin_5_fold/fold0", # 验证集不用增强
#     "test_dir" :"/home/liubo/data/graduate/resampled_classification_dataset/origin_5_fold/fold0",  # 测试在 predict中用到
#     "batch_size" : 8,
#     "log_dir":"./logs/"+ model_name+time_str,
#     "work_dir":"/home/liubo/nn_project/LungSystem/workdir/" + model_name+time_str,
#     "model_name": model_name,
#     "model_save_path":"/home/liubo/nn_project/LungSystem/models/guaduate/" + model_name,
#     "learn_rate":0.0001
# }

# 数据增强配置
model_name = "full_con_classifier_augu"
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
configs = {
    "train_dir_list":["/home/liubo/data/graduate/resampled_augumentation_classification_dataset/resample/fold1",
                      "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/resample/fold2",
                      "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/resample/fold3",
                      "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/resample/fold4"],
    "val_dir": "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/origin_5_fold/fold0", # 验证集不用增强
    "test_dir" :"/home/liubo/data/graduate/resampled_augumentation_classification_dataset/origin_5_fold/fold0",  # 测试在 predict中用到
    "batch_size" : 8,
    "log_dir":"./logs/"+ model_name+time_str,
    "work_dir":"/home/liubo/nn_project/LungSystem/workdir/" + model_name+time_str,
    "model_name": model_name,
    "model_save_path":"/home/liubo/nn_project/LungSystem/models/guaduate/" + model_name,
    "learn_rate":0.0001
}


def get_net(input_shape=(64, 64, 64, 1), load_weight_path=None, features=False) -> Model:
    """
    获取模型结构
    """
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(strides=(2, 1, 1), pool_size=(2, 1, 1), padding="same")(x)

    x = Convolution3D(64, (3, 3, 3), activation='relu', strides=(1, 1, 1), padding='same', name='conv1', )(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

    x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)

    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)

    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a', strides=(1, 1, 1))(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b', strides=(1, 1, 1), )(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)

    last64 = Convolution3D(64, (2, 2, 2), activation="relu", name="last_64")(x)

    out_class = Convolution3D(5, (1, 1, 1), activation="softmax", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)

    model = Model(inputs=inputs, outputs=out_class)

    if load_weight_path is not None:  
        model.load_weights(load_weight_path, by_name=False)

    model.compile(optimizer=SGD(lr=configs["learn_rate"], 
                                momentum=0.9, 
                                nesterov=True),
                  loss=categorical_crossentropy, 
                  metrics=[categorical_crossentropy, categorical_accuracy])

    model.summary()
    return model


def train(model_name,load_weight_path=None):
    """
    训练模型 
    :param model_name: 模型名称
    :param load_weight_path:之前的预训练模型
    """
    train_dir_list = configs["train_dir_list"]
    val_dir = configs["val_dir"]
    test_dir = configs["test_dir"]
    batch_size = configs["batch_size"]
    work_dir = configs["work_dir"]
    dataset= ClassificationDataset(train_dir_list,val_dir,test_dir,batch_size)
    dataset.prepare_train_val_dataset()
    train_data,val_data = dataset.get_train_val_dataset()
    train_gen = dataset.data_generator(batch_size=8,record_list=train_data,is_train_set=True)
    val_gen = dataset.data_generator(batch_size=8,record_list=val_data,is_train_set=False)

    model = get_net()
    # 每隔1轮保存一次模型

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
    model.fit_generator(generator=train_gen, 
                        steps_per_epoch=int(len(train_data)/batch_size), 
                        epochs=100, 
                        validation_data=val_gen,
                        validation_steps=int(len(val_data)/batch_size), 
                        callbacks=[checkpoint, 
                                   checkpoint_fixed_name, 
                                   TensorBoard(log_dir=configs["log_dir"])])


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
    

    
    