###################################
#   coding=utf-8
#   !/usr/bin/env python
#   __author__ = 'LXA'
#   ctime 2020.10.15
#   矩形区域内绘制椭圆和圆形
###################################
from matplotlib.patches import Ellipse, Circle, Rectangle
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
rctan = Rectangle((-1, -1), 2, 2, color='y', alpha=1.0)
ax.add_patch(rctan)

# # 第二个参数是边数，第三个是距离中心的位置
# polygon1 = patches.RegularPolygon((0, 0), 5, 1.25, color= "r")
# ax.add_patch(polygon1)
# # x1 = np.random.uniform(-1, 1, 3000) # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
# # y1 = np.random.uniform(-1, 1, 3000) # 随
# # plt.scatter(x1, y1, cmap='rainbow', alpha=0.25)

cir1 = Circle(xy=(0.1, 0.1), radius=0.125, color='w', alpha=1.0)
cir2 = Circle(xy=(0.3, 0.5), radius=0.075, color='w', alpha=1.0)
cir3 = Circle(xy=(0.6, 0.2), radius=0.15, color='w', alpha=1.0)
cir4 = Circle(xy=(0.825, 0.5), radius=0.075, color='w', alpha=1.0)
cir5 = Circle(xy=(0.1, 0.75), radius=0.1, color='w', alpha=1.0)
cir6 = Circle(xy=(-0.1, 0.8), radius=0.075, color='w', alpha=1.0)
cir7 = Circle(xy=(-0.4, 0.5), radius=0.075, color='w', alpha=1.0)
cir8 = Circle(xy=(-0.6, 0.2), radius=0.075, color='w', alpha=1.0)
cir9 = Circle(xy=(-0.8, 0.7), radius=0.075, color='w', alpha=1.0)
cir10 = Circle(xy=(-0.9, 0.1), radius=0.1, color='w', alpha=1.0)

cir11 = Circle(xy=(-0.1, -0.75), radius=0.1, color='w', alpha=1.0)
cir12 = Circle(xy=(-0.4, -0.8), radius=0.075, color='w', alpha=1.0)
cir13 = Circle(xy=(-0.3, -0.5), radius=0.075, color='w', alpha=1.0)
cir14 = Circle(xy=(-0.6, -0.2), radius=0.125, color='w', alpha=1.0)
cir15 = Circle(xy=(-0.825, -0.5), radius=0.075, color='w', alpha=1.0)
cir16 = Circle(xy=(0.1, -0.5), radius=0.075, color='w', alpha=1.0)
cir17 = Circle(xy=(0.3, -0.2), radius=0.105, color='w', alpha=1.0)
cir18 = Circle(xy=(0.5, -0.75), radius=0.125, color='w', alpha=1.0)
cir19 = Circle(xy=(0.725, -0.3), radius=0.1, color='w', alpha=1.0)
cir20 = Circle(xy=(0.9, -0.9), radius=0.075, color='w', alpha=1.0)
ax.add_patch(cir1)
ax.add_patch(cir2)
ax.add_patch(cir3)
ax.add_patch(cir4)
ax.add_patch(cir5)
ax.add_patch(cir6)
ax.add_patch(cir7)
ax.add_patch(cir8)
ax.add_patch(cir9)
ax.add_patch(cir10)
ax.add_patch(cir11)
ax.add_patch(cir12)
ax.add_patch(cir13)
ax.add_patch(cir14)
ax.add_patch(cir15)
ax.add_patch(cir16)
ax.add_patch(cir17)
ax.add_patch(cir18)
ax.add_patch(cir19)
ax.add_patch(cir20)
# ax.add_patch(ell1)

plt.axis('scaled')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length

plt.show()
