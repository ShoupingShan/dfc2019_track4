import numpy as np

format_float = []
numbers_float = []
number = '359'
point_name = 'xy_base.txt'
f = open(point_name, 'r')
data = f.readlines()  #txt中所有字符串读入data
for line in data:
    line = line[:]
    line = line.split(',')
    for l in line[0:2]:
         l = float(l)
         format_float.append(l)
    numbers_float.append(format_float)
    format_float = []
point = np.array(numbers_float)
numbers_float = []
f.close()
filename = 'extend_sample_DATA_2.txt'
g = open(filename, 'r')
data2 = g.readlines()  #txt中所有字符串读入data
for line in data2:
    line = line[:]
    line = line.split(',')
    for l in line:
         l = float(l)
         format_float.append(l)
    numbers_float.append(format_float)
    format_float = []
point_zir = np.array(numbers_float)

final_file = 'final2.txt'
h = open(final_file,'a')
for xy, zir in zip(point[:len(numbers_float)], point_zir ):
    h.write(str(xy[0]))
    h.write(',')
    h.write(str(xy[1]))
    h.write(',')
    h.write(str(zir[0]))
    h.write(',')
    h.write(str(zir[1]))
    h.write(',')
    h.write(str(zir[2]))
    h.write('\n')
h.close()
