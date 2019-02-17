import numpy as np

format_float = []
numbers_float = []
number = '359'
point_name = 'JAX_'+number+'_PC3.txt'
f = open('../data/dfc/inference_data/in/'+ point_name, 'r')
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
f.close()
filename = 'xy_base.txt'
g = open(filename, 'w')
for line in numbers_float:
    g.write(str(line[0]))
    g.write(',')
    g.write(str(line[1]))
    g.write('\n')
g.close()
