""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Show_mode = 0
showsz=1000
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True
cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')

def showpoints(xyz,c_gt=None, c_pred = None ,c_res=None,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True,ballradius=10):
    global showsz,mousex,mousey,zoom,changed
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
    def render():
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
            label = 'Ground Truth'
        elif Show_mode == 1:
            label = 'Predict Labels'
        else:
            label = 'Residual'
        cv2.putText(show, 'Mode: '+label, (showsz - 600, showsz - 60), 2, 1, (255, 255, 255))
    changed=True
    while True:
        if changed:
            render()
            changed=False
        cv2.imshow('show3d',show)
        if waittime==0:
            cmd=cv2.waitKey(10)%256
        else:
            cmd=cv2.waitKey(waittime)%256
        if cmd==ord('q'):
            break
        elif cmd==ord('Q'):
            sys.exit(0)

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
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png',show)
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
    number = '114'
    point_name = 'JAX_'+number+'_PC3.txt'
    point_cls = 'JAX_'+number+'_CLS.txt'
    # point_name = 'OMA_' + number + '_PC3.txt'
    # point_cls = 'OMA_' + number + '_CLS.txt'
    gt = []
    pred = []
    Residual = []
    f = open('/home/shp/Documents/Code/Python/Contest/dfc2019/track4/pointnet2/data/dfc/inference_data/validation/'+ point_name, 'r')
    data = f.readlines()  #txt中所有字符串读入data
    for line in data:
        line = line[:-1]
        line = line.split(',')
        for l in line:
            l = float(l)
            format_float.append(l)
        numbers_float.append(format_float)
        format_float = []
    point = np.array(numbers_float)
    numbers_float = []
    gt = None
    # print(np.unique(gt))
    p = open('/home/shp/Documents/Code/Python/Contest/dfc2019/track4/pointnet2/data/dfc/inference_data/val_softmax/'+point_cls ,'r')
    data3 = p.readlines()
    print("Pred:", np.unique(np.array(data3)))
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

    Residual= None

    print(np.unique(pred))
    showpoints(point[:,0:3], c_gt = gt, c_pred= pred, c_res= Residual,ballradius = 2, normalizecolor=False,showrot= True)


