import numpy as np
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

labels = ["Undefined", "Ground", "High Vegetation", "Building", "Water", "Bridge Deck"]
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels) - 1))
    plt.xticks(xlocations, labels[1:], rotation=90)
    plt.yticks(xlocations, labels[1:])
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label',fontsize=16)
tick_marks = np.array(range(len(labels) -1 )) + 0.5

confusion_matrix = [[53938439, 3508, 138471, 105089, 17631, 15570],
                             [20872, 11772602, 288309, 87, 1106, 13804],
                             [213420, 368051, 10462420, 27, 1542, 6766],
                             [51480, 103, 197, 1322731, 141, 1513],
                             [4912, 1408, 3336, 26, 947060, 2837]]
confusion_matrix = np.array(confusion_matrix)
confusion_matrix = np.array(confusion_matrix[:,:-1])
np.set_printoptions(precision=3)
x = confusion_matrix.sum(axis=1)[:, np.newaxis]
for i in range(len(x)):
    if not x[i]:
        x[i] = 1
cm_normalized = confusion_matrix.astype('float') / x
# print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)
ind_array = np.arange(len(labels) - 1)
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.001:
        plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=12, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.21)
plot_confusion_matrix(cm_normalized, title='OVERALL SCORES')
# show confusion matrix
plt.savefig('confusion_matrix_overall.png', format='png')
plt.show()