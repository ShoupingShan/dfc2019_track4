import argparse
import copy
from datetime import datetime
from enum import Enum
import glob
import importlib
import json
import logging
import math
import numpy as np
import os
import pickle
from pointset import PointSet
import pprint
from queue import Queue
import subprocess
import sys
import tempfile
import tensorflow as tf
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models')) # no, really model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
from numba import jit

class InputType(Enum):
    TXT='TXT'
    LAS='LAS'

class OutputType(Enum):
    LABELS='LABELS'
    LAS='LAS'
    BOTH='BOTH'
    
    def __str__(self):
        return self.value

def parse_args(argv):
    # Setup arguments & parse
    parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='pointSIFT_pointnet', help='Model name [default: pointnet2_sem_seg_two_feat]')
    parser.add_argument('--extra-dims', type=int, default=[3, 4], nargs='*', help='Extra dims')
    parser.add_argument('--model_path', default='/home/xaserver1/Documents/Contest/shp/pointsift/log/model/best_miou_model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during inference [default: 16]')
    parser.add_argument('--n_angles', type=int, default=4, help='Number of angles to use to sample image with')
    # parser.add_argument('--input_path', required=True, help='Input point clouds path')
    parser.add_argument('--input_path', type=str, default="/home/xaserver1/Documents/Contest/shp/post_process/data", help='Input point clouds path')
    parser.add_argument('--input_type', type=InputType, choices=list(InputType), default=InputType.TXT)
    # parser.add_argument('--output_path', required=True, help='Output path')
    parser.add_argument('--output_path', type=str, default="/home/xaserver1/Documents/Contest/shp/post_process/result", help='Output path')
    parser.add_argument('--output_type', type=OutputType, choices=list(OutputType), default=OutputType.LABELS)
    
    return parser.parse_args(argv[1:])

def start_log(opts):
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    rootLogger = logging.getLogger()

    logFormatter = logging.Formatter("%(asctime)s %(threadName)s[%(levelname)-3.3s] %(message)s")
    fileHandler = logging.FileHandler(os.path.join(opts.output_path,os.path.splitext(os.path.basename(__file__))[0]+'.log'),mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    logFormatter = logging.Formatter("%(threadName)s[%(levelname)-3.3s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.level=logging.DEBUG

    logging.debug('Options:\n'+pprint.pformat(opts.__dict__))


# Set global variables
FLAGS = parse_args(sys.argv)
start_log(FLAGS)

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
os.system('cp %s %s' % (MODEL_FILE, FLAGS.output_path)) # bkp of model def
os.system('cp '+__file__+' %s' % (FLAGS.output_path)) # bkp of train procedure
NUM_CLASSES = 6
COLUMNS = np.array([0,1,2]+FLAGS.extra_dims)
NUM_DIMENSIONS = len(COLUMNS)

#with open(os.path.join(os.path.dirname(FLAGS.model_path),'dfc_train_fold_5_metadata.pickle'),'rb') as f:
#    METADATA = pickle.load(f)
with open(os.path.join(os.path.dirname(FLAGS.model_path),'dfc_train_metadata.pickle'),'rb') as f:
    METADATA = pickle.load(f)
SCALE = np.sqrt(METADATA['variance'])
SCALE[0:3] = np.sqrt(np.mean(np.square(SCALE[0:3])))
LABEL_MAP = METADATA['decompress_label_map']

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def inference():
    # Generate list of files
    if FLAGS.input_type is InputType.TXT:
        files = glob.glob(os.path.join(FLAGS.input_path,"*.txt"))
    elif FLAGS.input_type is InputType.LAS:
        files = glob.glob(os.path.join(FLAGS.input_path,"*.las"))
    
    # Setup queues
    input_queue = Queue(maxsize=3)
    output_queue = Queue(maxsize=3)
    
    # Note: this threading implementation could be setup more efficiently, but it's about 2x faster than a non-threaded version.
    logging.info('Starting threads')
    pre_proc = threading.Thread(target=pre_processor,name='Pre-ProcThread',args=(sorted(files),input_queue))
    pre_proc.start()
    main_proc = threading.Thread(target=main_processor,name='MainProcThread',args=(input_queue,output_queue))
    main_proc.start()
    post_proc = threading.Thread(target=post_processor,name='PostProcThread',args=(output_queue,))
    post_proc.start()
    
    logging.debug('Waiting for threads to finish')
    pre_proc.join()
    logging.debug('Joined pre-processing thread')
    main_proc.join()
    logging.debug('Joined main processing thread')
    post_proc.join()
    logging.debug('Joined post-processing thread')
    
    logging.info('Done')


def prep_pset(pset):
    data64 = np.stack([pset.x,pset.y,pset.z,pset.i,pset.r],axis=1)
    offsets = np.mean(data64[:,COLUMNS],axis=0)
    data = (data64[:,COLUMNS]-offsets).astype('float32')
    
    n = len(pset.x)
    
    if NUM_POINT < n:
        ixs = np.random.choice(n,NUM_POINT,replace=False)
    elif NUM_POINT == n:
        ixs = np.arange(NUM_POINT)
    else:
        ixs = np.random.choice(n,NUM_POINT,replace=True)
    
    return data64[ixs,:], data[ixs,:] / SCALE[COLUMNS]


def get_batch(dataset, start_idx, end_idx):
    bsize = end_idx-start_idx
    rsize = min(end_idx,len(dataset))-start_idx
    batch_raw = np.zeros((rsize, NUM_POINT, 5), dtype=np.float64)
    batch_data = np.zeros((bsize, NUM_POINT, NUM_DIMENSIONS), dtype=np.float32)
    for i in range(rsize):
        pset = dataset[start_idx+i]
        batch_raw[i,...], batch_data[i,...] = prep_pset(pset)
    return batch_raw, batch_data


def pre_processor(files, input_queue):
    for file in files:
        
        logging.info('Loading {}'.format(file))
        pset = PointSet(file)
        psets = pset.split()
        num_batches = int(math.ceil((1.0*len(psets))/BATCH_SIZE))
        
        data = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            for k in range(FLAGS.n_angles):
                batch_raw, batch_data = get_batch(psets, start_idx, end_idx)
    
                if k == 0:
                    aug_data = batch_data
                else:
                    ang = (1.0*k)/(1.0*FLAGS.n_angles) * 2 * np.pi
                    if FLAGS.extra_dims:
                        aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:,:,0:3],angle=ang),
                                batch_data[:,:,3:]),axis=2)
                    else:
                        aug_data = provider.rotate_point_cloud_z(batch_data)
                
                data.append((batch_raw,aug_data))
        
        logging.debug('Adding {} to queue'.format(file))
        input_queue.put((pset,data))
        logging.debug('Added {} to queue'.format(file))
    logging.info('Pre-processing finished')
    input_queue.put(None)
    logging.debug('Pre-processing thread finished')


def main_processor(input_queue, output_queue):
    with tf.Graph().as_default():
        with tf.device('/device:GPU:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            logging.info("Loading model")
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}
        is_training = False
        logging.info("Model loaded")
    
        while True:
            in_data = input_queue.get()
            if in_data is None:
                break
            
            logging.info('Processing {}'.format(in_data[0].filename))
            batch_list = in_data[1]
            for k in range(len(batch_list)):
                batch_raw = batch_list[k][0]
                aug_data = batch_list[k][1]
                
                feed_dict = {ops['pointclouds_pl']: aug_data,
                            ops['is_training_pl']: is_training}
                pred_val = sess.run([ops['pred']], feed_dict=feed_dict)

                pred_val_prob = pred_val[0] # BxNx5
                pred_labels = np.argmax(pred_val[0], 2) # BxN
                
                # subset to true batch size as necessary
                if batch_raw.shape[0] != BATCH_SIZE:
                    pred_val_prob = pred_val_prob[0:batch_raw.shape[0], :]
                    pred_labels = pred_labels[0:batch_raw.shape[0], :]
                
                # Reshape pred_labels and batch_raw to (BxN,1) and (BxN,5) respectively (i.e. concatenate all point sets in batch together)
                pred_labels.shape = (pred_labels.shape[0]*pred_labels.shape[1])
                batch_raw.shape = (batch_raw.shape[0]*batch_raw.shape[1], batch_raw.shape[2])
                pred_val_prob.shape = (pred_val_prob.shape[0] * pred_val_prob.shape[1], pred_val_prob.shape[2])
                pred_val_prob = softmax(pred_val_prob)
                
                if k==0:
                    all_labels = pred_labels
                    all_points = batch_raw
                    all_val = pred_val_prob
                else:
                    # Concatenate all pointsets across all batches together
                    all_labels = np.concatenate((all_labels,pred_labels),axis=0)
                    all_points = np.concatenate((all_points,batch_raw),axis=0)
                    all_val = np.concatenate((all_val, pred_val_prob), axis=0)
            logging.debug('Adding {} to output queue'.format(in_data[0].filename))
            output_queue.put((in_data[0],all_points, all_labels, all_val))
            logging.debug('Added {} to output queue'.format(in_data[0].filename))
            input_queue.task_done()
        logging.info('Main processing finished')
        output_queue.put(None)
    logging.debug('Main processing thread finished')

@jit
def get_label_val(all_xyz, all_labels, all_val, original_xyz, K):
    original_label = np.zeros((original_xyz.shape[0]), dtype=np.int)
    class_label = np.arange(NUM_CLASSES)
    original_val = np.zeros((original_xyz.shape[0], NUM_CLASSES))
    for i in range(original_xyz.shape[0]):
        if i % 1000 is 0:
            print('i', i)
        dist = np.sum((all_xyz - original_xyz[i]) ** 2, axis=1)
        indK = np.argpartition(dist, K)[:K]
        class_cnt = np.zeros(NUM_CLASSES)
        for c in range(NUM_CLASSES):
            ind = np.where(all_labels[indK] == class_label[c])
            class_cnt[c] = len(ind[0])
        cls = class_label[np.argmax(class_cnt)]
        ind = np.where(all_labels[indK] == cls)
        original_label[i] = LABEL_MAP[cls]
        original_val[i, :] = np.sum(all_val[indK[ind]], axis=0) / np.max(class_cnt)
    return original_label, original_val



def post_processor(output_queue):
    while True:
        out_data = output_queue.get()
        if out_data is None:
            break

        K = 24
        pset = out_data[0]
        all_points = out_data[1]
        all_xyz = np.array(all_points[:, :3])
        all_labels = out_data[2]
        all_val = out_data[3]
        
        logging.info('Post-processing {}'.format(pset.filename))
        original_x = pset.x
        original_y = pset.y
        original_z = pset.z
        original_xyz = np.zeros((original_x.shape[0], 3), dtype=np.float)
        original_xyz[:, 0] = original_x
        original_xyz[:, 1] = original_y
        original_xyz[:, 2] = original_z

        original_label, original_val = get_label_val(all_xyz, all_labels, all_val, original_xyz, K)

        class_file = os.path.join(FLAGS.output_path, pset.filename + '_CLS.txt')
        probo_file = os.path.join(FLAGS.output_path, pset.filename + '_pred.txt')
        np.savetxt(class_file, original_label, fmt='%d')
        np.savetxt(probo_file, original_val, fmt='%5f, %5f, %5f, %5f, %5f, %5f', newline='\n')
        logging.debug('Finished {}'.format(pset.filename))
    logging.debug('Post-processing thread finished')


if __name__ == "__main__":
    inference()

    # import pydensecrf

