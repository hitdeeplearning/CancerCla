import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from classifiers.flat.resnet.liubo_2D_dataset import liubo_2D_dataset_c
from classifiers.flat.resnet.resnet import configs,get_net


model_path = "/home/liubo/nn_project/LungSystem/models/guaduate/resnet_v1_2D_classifier/resnet_v1_2D_classifier_best.hd5"

def predict():
    """
    进行预测
    """
    train_dir_list = configs["train_dir_list"]
    val_dir = configs["val_dir"]
    test_dir = configs["test_dir"]
    batch_size = configs["batch_size"]

    # 数据集
    dataset= liubo_2D_dataset_c(train_dir_list,val_dir,test_dir,batch_size)
    dataset.prepare_test_dataset()
    test_data = dataset.get_test_dataset()  # test_data : [test_img_list,test_label_list]

    # 模型
    model = get_net(load_weight_path=model_path)

    # 开始计时
    start_time = datetime.datetime.now()
    predIdxs = model.predict(test_data[0])
    # print('predIdxs:', predIdxs)
    # print(len(predIdxs))

    # 测试花费时间
    current_time = datetime.datetime.now()
    res = current_time - start_time
    print("Done in : ", res.total_seconds(), " seconds")
    print("model: " + model_path)
    print("label:")
    print(test_data[1].flatten())
    print("predict:")
    predict_label = np.argmax(predIdxs, axis=1)
    print(predict_label)
    # print(len(predict_label))

    # 计算预测的混淆矩阵
    confus_predict = confusion_matrix(test_data[1], predict_label)
    print("confusion matrix:")
    print(confus_predict)

    # 结果评估
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    classification_show = classification_report(test_data[1], predict_label, labels=None, target_names=target_names)
    print(classification_show)

if __name__ == "__main__":
    predict()
