import numpy as np
def softmax(pred_val):
    """Compute softmax values for each sets of scores in x."""
    val = np.array(pred_val)
    val.astype('float32')
    row_max = val.max(axis=2).reshape(val.shape[0], val.shape[1], 1)
    val = val - row_max
    val_exp = np.exp(val)
    val_exp_row_sum = val_exp.sum(axis=2).reshape(val.shape[0], val.shape[1], 1)
    val_softmax = val_exp/val_exp_row_sum
    return val_softmax
if __name__ == "__main__":
    pred_val=np.array([[[-77.26414,9.719617,-24.235611,-11.689369,-1.6125553,-14.184073],[-77.26414,9.719617,-24.235611,-11.689369,-1.6125553,-14.184073],[27.577152,-10.617909,2.8934019,2.2149427,-19.933443,-6.0476246]],[[-77.26414,9.719617,-24.235611,-11.689369,-1.6125553,14.184073],[77.26414,9.719617,-24.235611,-11.689369,-1.6125553,-14.184073],[-27.577152,-10.617909,2.8934019,2.2149427,-19.933443,-6.0476246]]])
    score = softmax(pred_val)
    pred = np.argmax(score, 2)
    pred_softmax = np.where(score >= 0.8)
    pred_softmax_1 = np.where(score >= 0.8, 1, -1)

    pred_softmax = np.argmax(pred_softmax_1, 2)
    labels = np.zeros(score.shape[:-1])
    print(score.shape)
    print(labels)

    print(score)
    print('Pred')
    print(pred)
    print('Pred Softmax')
    print(np.shape(pred_softmax))

    # labels[pred_softmax[0]][pred_softmax[1]][pred_softmax[2]] = 1
    print(labels)
    print(pred_softmax)


    print(score.sum(axis=2))