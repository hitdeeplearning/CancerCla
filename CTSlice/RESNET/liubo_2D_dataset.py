import os
import glob
import random
import cv2
import keras as K
import numpy as np


class liubo_2D_dataset_c:
    """
    2D分类数据集
    """
    def __init__(self, train_dir_list, val_dir,test_dir,batch_size):
        self.train_dir_list = train_dir_list
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size


    def get_class_label(self,sample_path):
        class_label = sample_path.split("/")[-2]
        if class_label == "zero":
            class_label = 0
        elif class_label == "one":
            class_label = 1
        elif class_label == "two":
            class_label = 2
        elif class_label == "three":
            class_label = 3
        elif class_label == "four":
            class_label = 4
        one_hot_labels = K.utils.to_categorical(class_label, num_classes=5)
        return one_hot_labels

    def prepare_train_val_dataset(self):

        # 训练集合
        train_dir_list = self.train_dir_list

        self.train_zero_samples = []
        self.train_one_samples = []
        self.train_two_samples = []
        self.train_three_samples = []
        self.train_four_samples = []
        for train_dir in train_dir_list:
            self.train_zero_samples += glob.glob(train_dir+'/zero/' + '*.png')
            self.train_one_samples += glob.glob(train_dir + '/one/' + "*.png")
            self.train_two_samples += glob.glob(train_dir + '/two/' + "*.png")
            self.train_three_samples += glob.glob(train_dir + '/three/' + "*.png")
            self.train_four_samples += glob.glob(train_dir + '/four/' + "*.png")

        self.train_origin_path = [self.train_zero_samples,
                                  self.train_one_samples,
                                  self.train_two_samples,
                                  self.train_three_samples,
                                  self.train_four_samples]
        print("-"* 20)
        print("train zero: ", len(self.train_zero_samples))
        print("train one: ", len(self.train_one_samples))
        print("train two: ", len(self.train_two_samples))
        print("train three: ", len(self.train_three_samples))
        print("train four: ", len(self.train_four_samples))
        print("-"* 20)

        # 测试集合
        val_dir = self.val_dir
        self.val_zero_samples = glob.glob(val_dir+'/zero/' + '*.png')
        self.val_one_samples = glob.glob(val_dir+'/one/' + '*.png')
        self.val_two_samples = glob.glob(val_dir+'/two/' + '*.png')
        self.val_three_samples = glob.glob(val_dir+'/three/' + '*.png')
        self.val_four_samples = glob.glob(val_dir+'/four/' + '*.png')

        self.val_origin_path = [self.val_zero_samples,
                                  self.val_one_samples,
                                  self.val_two_samples,
                                  self.val_three_samples,
                                  self.val_four_samples]

        print("-"* 20)
        print("val zero: ", len(self.val_zero_samples))
        print("val one: ", len(self.val_one_samples))
        print("val two: ", len(self.val_two_samples))
        print("val three: ", len(self.val_three_samples))
        print("val four: ", len(self.val_four_samples))
        print("-"* 20)

    def prepare_test_dataset(self):
        test_dir = self.test_dir
        self.test_zero_samples = glob.glob(test_dir+'/zero/' + '*.png')
        self.test_one_samples = glob.glob(test_dir + '/one/' + "*.png")
        self.test_two_samples = glob.glob(test_dir + '/two/' + "*.png")
        self.test_three_samples = glob.glob(test_dir + '/three/' + "*.png")
        self.test_four_samples = glob.glob(test_dir + '/four/' + "*.png")

        self.test_origin_path = [self.test_zero_samples,
                                 self.test_one_samples,
                                 self.test_two_samples,
                                 self.test_three_samples,
                                 self.test_four_samples]

        print("-"* 20)
        print("test zero: ", len(self.test_zero_samples))
        print("test one: ", len(self.test_one_samples))
        print("test two: ", len(self.test_two_samples))
        print("test three: ", len(self.test_three_samples))
        print("test four: ", len(self.test_four_samples))
        print("-"* 20)

    

    def get_train_val_dataset(self):
        """
        TODO 采样方式可以改进，现在先用个简单的
        这里使用随机过采样和随机欠采样组合的方式 , 采样后shuffle
        更多方式请看 https://www.cnblogs.com/wkslearner/p/8870673.html
        """
        train_dataset = []
        val_dataset = []
        for one_class_samples in self.train_origin_path:
            for sample_path in one_class_samples:
                img = cv2.imread(sample_path,cv2.IMREAD_GRAYSCALE)
                img = img[:,:,np.newaxis]
                label = self.get_class_label(sample_path)
                train_dataset.append([img,label])

        for one_class_samples in self.val_origin_path:
            for sample_path in one_class_samples:
                img = cv2.imread(sample_path,cv2.IMREAD_GRAYSCALE)
                img = img[:,:,np.newaxis]
                label = self.get_class_label(sample_path)
                val_dataset.append([img,label])

                
        random.shuffle(train_dataset)

        return train_dataset, val_dataset

    def get_test_dataset(self):
        test_img_list = []
        test_label_list = []
        for one_class_samples in self.test_origin_path:
            for sample_path in one_class_samples:
                img = cv2.imread(sample_path,cv2.IMREAD_GRAYSCALE)
                img = img[:,:,np.newaxis]
                class_label = sample_path.split("/")[-2]
                if class_label == "zero":
                    class_label = 0
                elif class_label == "one":
                    class_label = 1
                elif class_label == "two":
                    class_label = 2
                elif class_label == "three":
                    class_label = 3
                elif class_label == "four":
                    class_label = 4
                test_img_list.append(img)
                test_label_list.append(class_label)
        x = np.array(test_img_list)
        y = np.array(test_label_list)
        return [x,y]
    