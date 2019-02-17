""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Show_mode = 0
showsz=600
mousex,mousey=0.5,0.5
zoom=2.0
changed=True


#鼠标事件
def get_rect(im, title='show3d'):
    global changed
    changed = True
    # cv2.namedWindow(title)
    # cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None and (flags & cv2.EVENT_FLAG_LBUTTON):
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']


    TL = []
    BR = []

    im_draw_save = np.copy(im)
    while True:
        mouse_params = {'tl': None, 'br': None, 'current_pos': None,
                        'released_once': False}
        cv2.setMouseCallback(title, onMouse, mouse_params)
        cv2.imshow(title, im_draw_save)

        while mouse_params['br'] is None:
            im_draw_cur = np.copy(im_draw_save)
            if mouse_params['tl'] is not None:
                # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
                cv2.rectangle(im_draw_cur, mouse_params['tl'],
                    mouse_params['current_pos'], (255, 0, 0)) #bgr
            cv2.imshow(title, im_draw_cur)
            _ = cv2.waitKey(10)

        tl = [min(mouse_params['tl'][0], mouse_params['br'][0],showsz),
              min(mouse_params['tl'][1], mouse_params['br'][1],showsz)]
        br = [max(mouse_params['tl'][0], mouse_params['br'][0],0),
              max(mouse_params['tl'][1], mouse_params['br'][1],0)]
        TL.append(tl)
        BR.append(br)

        while 1:
            k = cv2.waitKey(10) & 0xFF
            if k == ord('s'):  # 继续框选
                im_draw_save = np.copy(im_draw_cur)
                print('继续框选')
                print('tl', TL)
                print('br', BR)
                break

            if k == ord('e'):  # 退出，标记下一样本
                print('保存并退出')
                print('tl', TL)
                print('br', BR)
                return TL, BR

            if k == ord('c'):  # 清除上一次结果,继续框选
                TL.pop()
                BR.pop()
                print('清除上一次结果，继续框选')
                print('tl', TL)
                print('br', BR)
                break

            if k == ord('d'):  # 清除上一次结果,直接退出
                TL.pop()
                BR.pop()
                print('清除上一次结果，并退出')
                print('tl', TL)
                print('br', BR)
                return TL, BR


dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')


def showpoints(xyz_all, title, root_dir, save_dir=None, c_gt=None, c_pred=None, c_res=None, label=None,
               waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0,0,0), normalizecolor=True, ballradius=10):
    global showsz,mousex,mousey,zoom,changed
    xyz = np.array(xyz_all[:,0:3])
    label_all = label

    M = np.max(xyz, axis=0)
    m = np.min(xyz, axis=0)
    r = (M-m) /showsz


    if c_gt is None:
        c0=np.zeros((len(xyz),),dtype='float32')+255
        c1=np.zeros((len(xyz),),dtype='float32')+255
        c2=np.zeros((len(xyz),),dtype='float32')+255
    else:
        c0=c_gt[:,0]
        c1=c_gt[:,1]
        c2=c_gt[:,2]

    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0

    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    rect_flag = True

    def render(title, root_dir, save_dir, rect_flag=rect_flag):
        show[:]=background

        l = len(xyz)
        for ind in range(l):
            x_ind = int((xyz[ind, 0] - m[0]) // r[0])
            y_ind = int((xyz[ind, 1] - m[1]) // r[1])
            show[x_ind, y_ind, 0] = c0[ind]
            show[x_ind, y_ind, 1] = c1[ind]
            show[x_ind, y_ind, 2] = c2[ind]

        if Show_mode==0:
            labels = 'Ground Truth'
        elif Show_mode == 1:
            labels = 'Predict Labels'
        else:
            labels = 'Residual'
        cv2.putText(show, 'Mode: '+labels, (showsz - 600, showsz - 60), 2, 1, (255, 255, 255))
        cv2.imshow(title, show)

        if(rect_flag):
            a, b = get_rect(show, title=title)  # 鼠标画矩形框
            print(a)
            print(b)
            num_point = 0
            collection_data = []
            collection_label = []

            if save_dir is None:
                save_dir = root_dir
            file_path = save_dir + title + '_PC3_extend.txt'
            label_path = save_dir + title + '_CLS_extend.txt'

            f = open(file_path, 'w')
            g = open(label_path, 'w')

            for index_x, index_y in zip(a,b):
                x_range1 = index_x[1] * r[0] + m[0]
                x_range2 = index_y[1] * r[0] + m[0]
                y_range1 = index_x[0] * r[1] + m[1]
                y_range2 = index_y[0] * r[1] + m[1]

                print('x_range1', x_range1)
                print('x_range2', x_range2)
                print('y_range1', y_range1)
                print('y_range2', y_range2)

                for items,l in zip(xyz_all, label_all):
                    if items[0] >= x_range1 and items[0] <= x_range2 and items[1] >= y_range1 and items[1] <= y_range2:
                        num_point += 1
                        collection_data.append(items[:])
                        collection_label.append(l)
            for data, label in zip(collection_data, collection_label):
                f.write(str(data[0]))
                f.write(',')
                f.write(str(data[1]))
                f.write(',')
                f.write(str(data[2]))
                f.write('\n')
                g.write(str(label))
                g.write('\n')
            f.close()
            g.close()
            print("%d points are selected." % num_point)
            print("Saved in %s." % file_path)

    changed = True
    rect_flag = False
    while True:
        if changed:
            render(title, root_dir, save_dir=save_dir, rect_flag=rect_flag)
            rect_flag = False  #no longer choose rect
            changed=False
        cv2.imshow(title, show)
        if waittime==0:
            cmd=cv2.waitKey(10)%256
        else:
            cmd=cv2.waitKey(waittime)%256
        if cmd==ord('n'):
            cv2.destroyWindow(title)
            break
        elif cmd==ord('q'):
            cv2.destroyWindow(title)
            sys.exit(0)
        elif cmd==ord('d'):
            rect_flag = True
            changed = True

        if cmd==ord('g') or cmd == ord('p') or cmd == ord('r'):
            if cmd == ord('g'):
                global Show_mode
                Show_mode = 0
                if c_gt is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_gt[:,0]
                    c1=c_gt[:,1]
                    c2=c_gt[:,2]
            elif cmd == ord('p'):
                Show_mode = 1
                if c_pred is None:
                    c0=np.zeros((len(xyz),),dtype='float32')+255
                    c1=np.zeros((len(xyz),),dtype='float32')+255
                    c2=np.zeros((len(xyz),),dtype='float32')+255
                else:
                    c0=c_pred[:,0]
                    c1=c_pred[:,1]
                    c2=c_pred[:,2]
            else:
                Show_mode = 2
                if c_res is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_res[:, 0]
                    c1 = c_res[:, 1]
                    c2 = c_res[:, 2]
            if normalizecolor:
                c0/=(c0.max()+1e-14)/255.0
                c1/=(c1.max()+1e-14)/255.0
                c2/=(c2.max()+1e-14)/255.0
            c0=np.require(c0,'float32','C')
            c1=np.require(c1,'float32','C')
            c2=np.require(c2,'float32','C')
            changed = True

    return cmd


if __name__=='__main__':
    np.random.seed(100)
    numbers_float = []
    format_float = []
    #r g b->g r b
    color_map = [[255, 255, 255],  [20, 97, 199], [34, 139, 34], [235, 206, 135], [211, 102, 160], [135, 138, 128]]
    # undefine, ground, high vegetation, building, water, Bridge Deck

    root_dir = '/home/jinyue/Track4/'
    save_dir = '/home/jinyue/Track4_extend/'
    predict = True
    pred_root = '/home/jinyue/Track4_pred/out_sift_gpu_1/'

    files_path = glob.glob(os.path.join(root_dir, "*_PC3.txt"))
    num = np.shape(files_path)[0]
    for ind in range(20):
        print('No.', ind)

        pc3_path = files_path[ind]
        cls_path = pc3_path[:-7] + 'CLS.txt'
        _, tempfilename = os.path.split(pc3_path)
        title = tempfilename[:-8]
        print('Title:', title)
        # cv2.namedWindow(title)

        if predict:
            pred_path = pred_root + tempfilename[:-7] + 'CLS.txt'
        else:
            pred_path = cls_path

        gt = []
        pred = []
        Residual = []

        f = open(pc3_path, 'r')
        data = f.readlines()  #txt中所有字符串读入data
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

        g = open(cls_path, 'r')
        data2 = g.readlines()
        print("GT:", np.unique(np.array(data2)))
        label = []
        numbers_float = []
        format_float = []
        for line in data2:
            line = int(line)
            label.append(line)
            if line == 0:
                format_float.append(color_map[0])
            elif line ==2:
                format_float.append(color_map[1])
            elif line ==5:
                format_float.append(color_map[2])
            elif line ==6:
                format_float.append(color_map[3])
            elif line ==9:
                format_float.append(color_map[4])
            elif line == 17:
                format_float.append(color_map[5])
            numbers_float.append(format_float)
            format_float = []

        label = np.array(label)
        gt = np.array(numbers_float)
        gt = np.reshape(gt, [len(point), 3])
        # print(np.unique(gt))
        numbers_float = []
        g.close()

        p = open(pred_path, 'r')
        data3 = p.readlines()
        print("Pred:", np.unique(np.array(data3)))
        pred = []
        numbers_float = []
        format_float = []
        for line in data3:
            line = int(line)
            if line == 0:
                format_float.append(color_map[0])
            elif line ==2:
                format_float.append(color_map[1])
            elif line ==5:
                format_float.append(color_map[2])
            elif line ==6:
                format_float.append(color_map[3])
            elif line ==9:
                format_float.append(color_map[4])
            elif line == 17:
                format_float.append(color_map[5])
            numbers_float.append(format_float)
            format_float = []
        pred = np.array(numbers_float)
        pred = np.reshape(pred,[len(point), 3])
        numbers_float = []
        format_float = []
        p.close()
        for line_gt, line_pred in zip(data2,data3):
            line_gt = int(line_gt)
            line_pred = int(line_pred)
            if line_gt - line_pred is not 0 and line_gt is not 0:
                format_float.append([0, 0, 255]) #Red color
            else:
                format_float.append([0,0,0])
            numbers_float.append(format_float)
            format_float = []
        Residual = np.array(numbers_float)
        Residual = np.reshape(Residual, [len(point), 3])
        # print(label)
        showpoints(point, title, root_dir, save_dir=save_dir, c_gt=gt, label=label, c_pred=pred, c_res=Residual,
                   ballradius=2, normalizecolor=False, showrot=False)


