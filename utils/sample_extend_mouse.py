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



# cv2.moveWindow('show3d',0,0)
# cv2.setMouseCallback('show3d',onmouse)

dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')


def showpoints(xyz_all, title, root_dir, save_dir=None, c_gt=None, c_pred=None, c_res=None, label=None,
               waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0,0,0), normalizecolor=True, ballradius=10):
    global showsz,mousex,mousey,zoom,changed
    xyz = xyz_all[:,0:3]
    label_all = label
    xyz=xyz-xyz.mean(axis=0)
    radius=((xyz**2).sum(axis=-1)**0.5).max()
    xyz/=(radius*2.2)/showsz
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
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))
        if not freezerot:
            yangle=(mousex-0.5)*np.pi*1.2
        else:
            yangle=0
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(80,showsz-30),0,1,(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(80,showsz-70),0,1,(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(80,showsz-100),0,1,(255,0,0))
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
            a = np.array(a)- showsz/2
            b = np.array(b) - showsz/2
            # print(a)
            num_point = 0
            collection_data = []
            collection_label = []

            # file_name = 'extend_sample_DATA_1.txt'
            # label_name= 'extend_sample_GT_1.txt'
            if save_dir is None:
                save_dir = root_dir
            file_path = save_dir + title + '_PC3_extend.txt'
            label_path = save_dir + title + '_CLS_extend.txt'

            f = open(file_path, 'w')
            g = open(label_path, 'w')

            for index_x, index_y in zip(a,b):
                for items,l in zip(xyz_all, label_all):
                    if items[0] >= index_x[0] and items[0] <= index_y[0] and items[1] >= index_x[1] and items[1] <= index_y[1]:
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

        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=2.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png', show)
        if waittime!=0:
            break
    return cmd


if __name__=='__main__':
    np.random.seed(100)
    numbers_float = []
    format_float = []
    #r g b->g r b
    color_map = [[255, 255, 255],  [133, 205, 63], [255, 0, 0], [206, 135, 250], [112, 147, 219], [125, 139, 107]]
    # undefine, ground, high vegetation, building, water, Bridge Deck

    root_dir = '/home/shp/Documents/Code/Python/Contest/dfc2019/track4/pointnet2/data/dfc/inference_data/Extend_path/in/'
    save_dir = '/home/shp/Documents/Code/Python/Contest/dfc2019/track4/pointnet2/data/dfc/inference_data/Extend_path/Mouse_extend/'

    files_path = glob.glob(os.path.join(root_dir, "*_PC3.txt"))
    num = np.shape(files_path)[0]
    predict = False
    work_checkpoint = 60
    for ind in range(num):
        ind = ind + work_checkpoint
        print('No.', ind)
        pc3_path = files_path[ind]
        cls_path = pc3_path[:-19] + '/gt/'+pc3_path[-15:-7]+ 'CLS.txt'
        _, tempfilename = os.path.split(pc3_path)
        title = tempfilename[:-8]
        print('Title:', title)
        # cv2.namedWindow(title)

        if predict:
            pred_path = pc3_path[:-7] + 'Pred.txt'
        else:
            pred_path = pc3_path[:-19] + '/out_sift_gpu_1/'+pc3_path[-15:-7]+ 'CLS.txt'

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
            if line_gt - line_pred is not 0:
                format_float.append([0, 255, 0]) #Red color
            else:
                format_float.append([0,0,0])
            numbers_float.append(format_float)
            format_float = []
        Residual = np.array(numbers_float)
        Residual = np.reshape(Residual, [len(point), 3])
        # print(label)
        showpoints(point, title, root_dir, save_dir=save_dir, c_gt=gt, label=label, c_pred=pred, c_res=Residual,
                   ballradius=2, normalizecolor=False, showrot=False)