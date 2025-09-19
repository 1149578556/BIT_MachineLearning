import csv
from sklearn.neighbors import KDTree

# 读取数据
data_points = []
#with open('Works/第3章k近邻法/LearningData.csv', 'r', encoding='utf-8') as f: # 打开文件夹为BIT_MachineLearning时
with open('LearningData.csv', 'r', encoding='utf-8') as f: # 打开当前文件夹时，可以直接使用IDLE运行
    reader = csv.reader(f)
    for row in reader:
        try:
            # 取第二列和第三列（索引1和2）
            x, y = float(row[1]), float(row[2])
            data_points.append([x, y])
        except (ValueError, IndexError):
            continue  # 跳过无效行

if not data_points:
    print("数据为空或格式错误。")
    exit(1)

# 构建KD树
tree = KDTree(data_points)

# 用户输入新点
try:
    user_x = float(input("请输入新点的x坐标："))
    user_y = float(input("请输入新点的y坐标："))
except ValueError:
    print("输入格式错误。")
    exit(1)

# 查询最近的3个点
dist, idx = tree.query([[user_x, user_y]], k=3)
print("距离最近的3个点为：")
for i, index in enumerate(idx[0]):
    print(f"第{i+1}个点: {data_points[index]}, 距离: {dist[0][i]:.4f}")