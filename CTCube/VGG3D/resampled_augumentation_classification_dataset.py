"""
一些预先说明：

原来的训练测试的目录在 /home/liubo/data/graduate/classification_dataset
现在将之前的数据合并在一起在进行重新采样
合并在一起的数据集保存在 /home/liubo/data/graduate/resampled_classification_dataset/total
total 中                     增强后
zero  19 + 6 = 25            25 * 6 = 150
one   596 + 30 = 626         626 * 6 = 3756
two   166 + 15 = 181         181*6 = 1086
three 24 + 6 = 30            30 *6 = 180
four  74 + 10 = 84           84 *6  = 504


本脚本说明：
脚本分三部
1. 用total 分成 五折 classify_to_5_fold
2. 进行数据增强 augumentation （翻转，旋转，噪声）
2. 重采样 resample 400张 * 5类 * 5折

"""
import os
import glob
import random
import cv2
import keras as K
import numpy as np
from tqdm import tqdm 
import shutil

config = {}
config["total_dir"] = "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/total"
config["augumentation_dir"] = "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/augumentation"
config["resample_dir"] = "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/resample"
config["origin_5_fold"] = "/home/liubo/data/graduate/resampled_augumentation_classification_dataset/origin_5_fold"
config["fold_name_list"] = ["fold0","fold1","fold2","fold3","fold4"]
config["class_name_list"] = ["zero","one","two","three","four"]

def classify_to_5_fold():

    # 创建目录
    origin_5_fold = config["origin_5_fold"]
    fold_name_list = config["fold_name_list"]
    class_name_list = config["class_name_list"]
    if os.path.exists(origin_5_fold): # 如果有之前的结果就先删除再创建
        shutil.rmtree(origin_5_fold)
        os.mkdir(origin_5_fold)
    else:
        os.mkdir(origin_5_fold)
    for f_name in fold_name_list:
        fold_path = origin_5_fold + "/" + f_name
        os.mkdir(fold_path)
        for c_name in class_name_list:
            fold_class_path = fold_path + "/" + c_name
            os.mkdir(fold_class_path)



    ## 尽量平均拷贝原始文件（可能原始数据量不能被5整除）
    total_dir = config["total_dir"]
    class_name_list = config["class_name_list"]
    fold_name_list = config["fold_name_list"]
    for c_name in class_name_list:
        c_samples = glob.glob(total_dir+"/"+ c_name +"/" + '*.png')
        for i in tqdm(range(len(c_samples)),desc="copy class " +c_name+" sample(total: " + str(len(c_samples))+") "):
            ori_path = c_samples[i]
            des_path = origin_5_fold + "/" + fold_name_list[i%5] + "/" + c_name
            shutil.copy(ori_path,des_path)
    

    ## 打印
    for f_name in fold_name_list:
        fold_path = origin_5_fold + "/" + f_name
        print(f_name+"+"*20)
        for c_name in class_name_list:
            fold_class_path = fold_path + "/" + c_name
            count = len(glob.glob(fold_class_path +"/"+ "*.png"))
            print(c_name + ":"+str(count))


    


# 每个图片是由多个小图片连在一起的，这个函数把小图像分割出来存进列表并返回
def load_cube_img(src_path, rows, cols, size):
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
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]
    return res

# 图像旋转数据增强
def rotation(cube,rows,cols,size):
    new_cube = np.zeros((rows * cols, size, size))
    angle = random.random()*180
    for i in range(len(cube)):
        img = cube[i]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        new_img = cv2.warpAffine(img,M,(size,size))
        new_cube[i] = new_img
    return new_cube

# 剪切
def move(cube,rows,cols,size):
    new_cube = np.zeros((rows * cols, size, size))
    move_percent = 0.2 # 左右平移最多移动20% 上下移动最多平移20%
    delta_x = random.ramdom() * move_percent * rows
    delta_y = random.ramdom() * move_percent * cols
    move_right =  1 if ramdom.random() > 0.5 else 0
    move_down = 1 if ramdom.random() > 0.5 else 0
    
    M = np.float32([
        [move_right, 1-move_right, delta_x],  # x轴方向上的平移
        [1-move_down, move_down, delta_y]   # y轴方向上的平移
        ])
    for i in range(len(cube)):
        img = cube[i]
        new_img = cv2.warpAffine(img,M,(size,size))
        new_cube[i] = new_img 
    return new_cube

def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def sp_noise(image,prob=0.1):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# 噪声
def make_g_noise(cube,rows,cols,size):
    new_cube = np.zeros((rows * cols, size, size))
    for i in range(len(cube)):
        img = cube[i]
        new_img = gasuss_noise(img)
        new_cube[i] = new_img 
    return new_cube

def make_sp_noise(cube,rows,cols,size):
    new_cube = np.zeros((rows * cols, size, size))
    for i in range(len(cube)):
        img = cube[i]
        new_img = sp_noise(img)
        new_cube[i] = new_img 
    return new_cube

def save_to_pic(des_path,rows,cols,size,cube):
    img = np.zeros((rows*size,cols*size))
    for row in range(rows):
        for col in range(cols):
            src_y = row * size
            src_x = col * size
            img[src_y:src_y+size,src_x:src_x+size] = cube[row*col+col]
    cv2.imwrite(des_path, img)



def augumentation():
    # 创建增强目录
    origin_5_fold = config["origin_5_fold"]
    augumentation_dir = config["augumentation_dir"]
    fold_name_list = config["fold_name_list"]
    class_name_list = config["class_name_list"]
    if os.path.exists(augumentation_dir): # 如果有之前的结果就先删除再创建
        shutil.rmtree(augumentation_dir)
        os.mkdir(augumentation_dir)
    else:
        os.mkdir(augumentation_dir)
    for f_name in fold_name_list:
        fold_path = augumentation_dir + "/" + f_name
        os.mkdir(fold_path)
        for c_name in class_name_list:
            fold_class_path = fold_path + "/" + c_name
            os.mkdir(fold_class_path)


    # 增强并拷贝
    for f_name in fold_name_list:
        for c_name in class_name_list:
            samples = glob.glob(origin_5_fold +"/"+ f_name+"/"+c_name+"/"+"*.png")
            for i in tqdm(range(len(samples)),desc="augumentation " +f_name +" "+c_name+" sample(total: " + str(len(samples))+") "):
                ori_path = samples[i]
                ori_name = ori_path.split("/")[-1].split(".")[0]
                des_path = augumentation_dir + "/" + f_name + "/" + c_name

                # 翻转
                origin = load_cube_img(ori_path, 8, 8, 64) 
                filp_lr = origin[:,:,::-1]
                filp_ud = origin[:,::-1,:]
                # 旋转 
                trans = rotation(origin,8,8,64)
                # 噪声
                gasuss = make_g_noise(origin,8,8,64)
                sp = make_sp_noise(origin,8,8,64)

                or_des_path = des_path + "/" + ori_name + "_or" + ".png"
                lr_des_path = des_path + "/" + ori_name + "_lr" + ".png"
                ud_des_path = des_path + "/" + ori_name + "_ud" + ".png"
                tr_des_path = des_path + "/" + ori_name + "_tr" + ".png"
                ga_des_path = des_path + "/" + ori_name + "_ga" + ".png"
                sp_des_path = des_path + "/" + ori_name + "_sp" + ".png"
                
                save_to_pic(or_des_path,8,8,64,origin)
                save_to_pic(lr_des_path,8,8,64,filp_lr)
                save_to_pic(ud_des_path,8,8,64,filp_ud)
                save_to_pic(tr_des_path,8,8,64,trans)
                save_to_pic(ga_des_path,8,8,64,gasuss)
                save_to_pic(sp_des_path,8,8,64,sp)
                

                 



    ## 打印
    for f_name in fold_name_list:
        fold_path = augumentation_dir + "/" + f_name
        print(f_name+"+"*20)
        for c_name in class_name_list:
            fold_class_path = fold_path + "/" + c_name
            count = len(glob.glob(fold_class_path +"/"+ "*.png"))
            print(c_name + ":"+str(count))


def resample():
    """
    凑齐 5折交叉验证 400张 * 5类 * 5折
    """

    # 创建文件夹
    resample_dir = config["resample_dir"]
    augumentation_dir = config["augumentation_dir"]
    fold_name_list = config["fold_name_list"]
    class_name_list = config["class_name_list"]

    if os.path.exists(resample_dir): # 如果有之前的结果就先删除再创建
        shutil.rmtree(resample_dir)
        os.mkdir(resample_dir)
    else:
        os.mkdir(resample_dir)
    for f_name in fold_name_list:
        fold_path = resample_dir + "/" + f_name
        os.mkdir(fold_path)
        for c_name in class_name_list:
            fold_class_path = fold_path + "/" + c_name
            os.mkdir(fold_class_path)


    for f_name in fold_name_list:
        for c_name in class_name_list:
            source_path = augumentation_dir + "/" + f_name+"/"+c_name + "/" +"*.png"
            source_path_file_list = glob.glob(source_path)
            sample_number = [random.randint(0,len(source_path_file_list)-1) for n in range(400)]
            target_dir = resample_dir + "/" + f_name +"/"+c_name
            for k in tqdm(range(len(sample_number)),desc= "copy to " + target_dir):
                augumentation_name = source_path_file_list[sample_number[k]].split("/")[-1]
                resample_name = augumentation_name.split(".")[0] + "_%04d" % k + ".png"
                shutil.copy(source_path_file_list[sample_number[k]],target_dir+"/"+resample_name)

    ## 打印
    for f_name in fold_name_list:
        fold_path = resample_dir + "/" + f_name
        print(f_name+"+"*20)
        for c_name in class_name_list:
            fold_class_path = fold_path + "/" + c_name
            count = len(glob.glob(fold_class_path +"/"+ "*.png"))
            print(c_name + ":"+str(count))

        

if __name__ == "__main__":
    # classify_to_5_fold()
    # augumentation()
    # resample()
    