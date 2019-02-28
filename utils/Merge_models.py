import numpy as np
import glob
import os

def Load_files(input_path): #读取所有文件名
    files = glob.glob(os.path.join(input_path,"*.txt"))
    return sorted(files)
def Load_data_from_file(files): #载入所有文件
    data = []
    for i in range(len(files)):
        data.append(np.loadtxt(files[i], dtype='uint8'))
    return data
def Compare(result_1, result_2):
    assert (len(result_1) == len(result_2))
    refer = []
    pred = []
    for i in range(len(result_1)):
        refer_items = np.zeros([len(result_1[i]),1],dtype='uint8')
        pred_items = np.zeros([len(result_1[i]),1],dtype='int8')

        for j in range(len(result_1[i])):
            if result_1[i][j] - result_2[i][j] is not 0:
                refer_items[j] = 1
                pred_items[j] = -1
            else:
                pred_items[j] = result_1[i][j]
        refer.append(refer_items)
        pred.append(pred_items)
    return refer, pred
def refer_compare(refer, pred, result_3):
    for i in range(len(refer)):
        for j in range(len(refer[i])):
            if refer[i][j]:
                pred[i][j] = result_3[i][j]
    return pred
def Save(pred, files, out_path):
    for i in range(len(pred)):
        f = open(out_path + files[i], 'w')
        pred_item = np.array(pred[i])
        for item in pred_item:
            f.write(str(item))
            f.write('\n')
        f.close()

def Merge_model():
    input_path1 = ' '
    input_path2 = ' '
    input_path3 = ' '
    out_path = ' '
    model1_data = Load_data_from_file(Load_files(input_path1))
    model2_data = Load_data_from_file(Load_files(input_path2))
    model3_data = Load_data_from_file(Load_files(input_path3))
    refer, pred = Compare(result_1 = model1_data, result_2 =model2_data)
    preds = refer_compare(refer, pred, result_3 = model3_data)
    output_file = Load_files(input_path1)[:][:-15]
    Save(preds, output_file, out_path)


if __name__ == "__main__":
    Merge_model()