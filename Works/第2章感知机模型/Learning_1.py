# 取定初值以及定义函数
import numpy as np
w = np.array([0, 0])  # 初始化权重向量
b = 0                 # 初始化偏置

def f(x):
    if w[0]*x[0] + w[1]*x[1] + b >= 0:
        return 1
    else:
        return -1

# 读取数据，注意**【文件路径】**
#file = open('Works/第2章感知机模型/LearningData.csv', 'r', encoding='utf-8') #当打开文件夹为BIT_MachineLearning时使用
file = open('LearningData.csv', 'r', encoding='utf-8')

# 第一组数据存储到 data1
data1 = []
file.readline()  # 跳过表头
line = file.readline().strip().split(',')
while len(line) > 2 and line[1] != '' and line[2] != '':
    # 读取前两列数据作为正类样本
    data1.append([eval(line[1]), eval(line[2])])
    line = file.readline().strip().split(',')

# 第二组数据存储到 data2
data2 = []
file.seek(0)  # 文件指针回到开头
file.readline()  # 跳过表头
line = file.readline().strip().split(',')
while len(line) > 6 and line[5] != '' and line[6] != '':
    # 读取第六、七列数据作为负类样本
    data2.append([eval(line[5]), eval(line[6])])
    line = file.readline().strip().split(',')

i, j = 0, 0
while True:
    updated = False
    # 遍历 data1，检查正类样本是否被正确分类
    for arr in data1:
        if f(arr) != 1:
            # 如果分类错误，更新权重和偏置
            w += np.array(arr)
            b += 1
            updated = True
            break #跳出 for 循环，重新开始检查
    if updated:
        # 如果已经更新，跳过检查 data2，重新开始
        continue
    # 遍历 data2，检查负类样本是否被正确分类
    for arr in data2:
        if f(arr) != -1:
            # 如果分类错误，更新权重和偏置
            w -= np.array(arr)
            b -= 1
            updated = True
            break #跳出 for 循环，重新开始检查
    if not updated:
        # 如果没有任何更新，说明所有样本都被正确分类，退出循环
        break

# 输出最终模型参数
print("最终得到的模型为：")
print("权重 w =", w)
print("偏置 b =", b)
print("分类函数为：f(x) = sign(", w[0], "*x(1)+", w[1], "*x(2)+", b, ")")
