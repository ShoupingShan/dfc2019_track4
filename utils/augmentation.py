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
    root_dir = '/home/jinyue/Track4_extend/'
    save_dir = '/home/jinyue/Track4_augment/'
    augment_num = 2
    rotate_fac = 1
    jitter_fac = 1
    sigma = 0.01
    clip = 0.05

    files_path = glob.glob(os.path.join(root_dir, "*_PC3_extend.txt"))
    num = np.shape(files_path)[0]
    for i in range(augment_num):
        print("No.", i)
        ind = random.randint(0, num-1)
        pc3_path = files_path[ind]
        cls_path = pc3_path[:-14] + 'CLS_extend.txt'

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
        extend_PC3_path = save_dir + str(i) + '_PC3_extend.txt'
        np.savetxt(extend_PC3_path, point, fmt='%f', delimiter=',')

        extend_CLS_path = save_dir + str(i) + '_CLS_extend.txt'
        shutil.copyfile(cls_path, extend_CLS_path)








