import numpy as np
import glob
import os
import random
import shutil


###########
# 绕Z轴旋转 #
###########
# point: vector(1*3:x,y,z)
# rotation_angle: scaler 0~2*pi
def rotate_point(point, rotation_angle):
    point = np.array(point)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                [-sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
    return rotated_point


# point = np.array([1,2,3])
# rotated_point = rotate_point(point, 0.1*np.pi)
# print rotated_point


###########
# 在XYZ上加高斯噪声 #
###########
def jitter_point(point, sigma=0.01, clip=0.05):
    assert (clip > 0)
    point = np.array(point)
    point = point.reshape(-1, 3)
    Row, Col = point.shape
    jittered_point = np.clip(sigma * np.random.randn(Row, Col), -1 * clip, clip)
    jittered_point += point
    return jittered_point


# jittered_point = jitter_point(point)
# print jittered_point


###########
# Data Augmentation #
###########
def augment_data(point, rotation_angle, sigma, clip):
    return jitter_point(rotate_point(point, rotation_angle), sigma, clip)


if __name__=="__main__":
    root_dir = '/home/xaserver1/Documents/Contest/shp/Track4/dfc2019_track4/data/Mouse_extend/'
    save_dir = '/home/xaserver1/Documents/Contest/shp/Track4/dfc2019_track4/data/Mouse_extend/Data_augment/'
    augment_num = 2
    rotate_fac = 1
    jitter_fac = 0
    sigma = 0.01
    clip = 0.05
    extend_fac = 2

    files_path = glob.glob(os.path.join(root_dir, "*PC3*.txt"))
    num = np.shape(files_path)[0]

    for i in range(extend_fac):
        for ind in range(num):
            print("No.", i*num + ind)
            # ind = random.randint(0, num-1)
            pc3_path = files_path[ind]
            cls_path = pc3_path.replace('PC3','CLS')
            # title = pc3_path[:8]
            _, PC3_filename = os.path.split(pc3_path)
            _, CLS_filename = os.path.split(cls_path) 

            # load data
            f = open(pc3_path, 'r')
            data = f.readlines()  # txt中所有字符串读入data
            numbers_float = []
            format_float = []
            for line in data:
                line = line[:]
                line = line.split(',')
                for l in line:
                    l = float(l)
                    format_float.append(l)
                numbers_float.append(format_float)
                format_float = []

            point = np.array(numbers_float)
            numbers_float = []
            f.close()

            # rotate
            r = random.random()
            if r < rotate_fac:
                rotate_angle = random.random() * 2 * np.pi
                point[:, :3] = rotate_point(point[:, :3], rotate_angle)

            # jitter
            r = random.random()
            if r < jitter_fac:
                point[:, :3] = jitter_point(point[:, :3], sigma, clip)

            # save data
            extend_PC3_path = save_dir + PC3_filename[:-4] +'_' + str(i) + '_augment.txt'
            # extend_PC3_path = save_dir + 'PC3_'+ str(ind) + '_augment.txt'

            np.savetxt(extend_PC3_path, point, fmt='%f', delimiter=',')

            extend_CLS_path = save_dir + CLS_filename[:-4] + '_'+ str(i) + '_augment.txt'
            # extend_CLS_path = save_dir + "CLS_" + str(ind) + '_augment.txt'
            shutil.copyfile(cls_path, extend_CLS_path)








