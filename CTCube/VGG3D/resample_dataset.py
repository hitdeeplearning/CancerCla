import os
import glob
import random
import cv2
import keras as K
import numpy as np

config = {
    "mean_pixel_values":118 
}

class ClassificationDataset:
    """
    数据集 功能：
    - 接受训练 测试数据地址
    - 准备训练验证集合数据 
    - 准备测试集合数据
    - 产生训练验证generator
    - 产生测试generator
    """
    def __init__(self, train_dir_list, val_dir,test_dir,batch_size):
        self.train_dir_list = train_dir_list
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size

    def get_class_label(self,sample_path):
        class_label = sample_path.split("/")[-1].split('_')[1]
        if class_label == 'SCLC':
            return "0"
        else:
            return class_label


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
                label = int(self.get_class_label(sample_path))
                train_dataset.append([sample_path,label])

        for one_class_samples in self.val_origin_path:
            for sample_path in one_class_samples:
                label = int(self.get_class_label(sample_path))
                val_dataset.append([sample_path,label])

                
        random.shuffle(train_dataset)

        return train_dataset, val_dataset

    def get_test_dataset(self):
        test_img_list = []
        test_label_list = []
        for one_class_samples in self.test_origin_path:
            for sample_path in one_class_samples:
                img = self.load_cube_img(sample_path)
                img3d = self.prepare_image_for_net3D(img)
                label = int(self.get_class_label(sample_path))
                test_img_list.append(img3d)
                test_label_list.append(label)

        x = np.vstack(test_img_list)
        y = np.vstack(test_label_list)
        return [x,y]


    # 每个图片是由多个小图片连在一起的，这个函数把小图像分割出来存进列表并返回
    def load_cube_img(self,src_path, rows=8, cols=8, size=64): 
        """
        每张图有8*8个小图,每一个小图 大小64* 64
        """
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        assert rows * size == img.shape[0]
        assert cols * size == img.shape[1]
        res = np.zeros((rows * cols, size, size))

        img_height = size   
        img_width = size    

        for row in range(rows):    
            for col in range(cols):  
                src_y = row * img_height
                src_x = col * img_width
                # 从上到下叠加在了一起
                res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]
        return res

    def prepare_image_for_net3D(self,img):
        """
        输入网络前进行类型转换 以及归一化和reshape
        """
        img = img.astype(np.float32)
        img -= config["mean_pixel_values"]
        img /= 255.
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
        return img

    def data_generator(self,batch_size, record_list, is_train_set=True):
        """
        数据生成器 每次返回一个batch的数据
        """
        batch_count = 0 

        while True:
            img_list = []
            class_list = []
            for record in record_list:
                #这样进行batch生成会导致未被样本整除的余数被舍弃
                class_label = int(record[1])
                cube_image = self.load_cube_img(record[0], 8, 8, 64)

        
                if is_train_set:  
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[::-1, :, :]  # 上下交换
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]  # 前后交换
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]  # 左右交换
                
                img3d = self.prepare_image_for_net3D(cube_image)
                img_list.append(img3d)
                class_list.append(class_label)
                batch_count += 1

                # 当累计到一个batch返回数据
                if batch_count >= batch_size:
                    x = np.vstack(img_list)  # 垂直方向累数据
                    y_class = np.vstack(class_list)
                    one_hot_labels = K.utils.to_categorical(y_class, num_classes=5)
                    yield x, {"out_class": one_hot_labels}
                    img_list = []
                    class_list = []
                    batch_count = 0
        


        
