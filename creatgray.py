import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 长方形参数
rectangle_width = 10  # 长方形的宽度
rectangle_height = 50  # 长方形的长度


X,Y = [],[]
# 在区间内绘制10个长方形
for i in range(10):
    rect_center_x = 1 + i  # 每个长方形中心的x坐标均匀分布在区间 [1, 10]
    rect_center_y = np.sin(rect_center_x)  # 长方形中心的y坐标
    X.append(rect_center_x)
    Y.append(rect_center_y)
def plotrec(X,Y):
    for (rect_center_x,rect_center_y) in zip(X,Y):
        # 计算切线方向
        dy_dx = np.cos(rect_center_x/30)/3 # 在该点的切线的斜率

        # 切线的方向角
        angle = np.arctan(dy_dx)

        # 计算长方形的旋转角度
        angle_degrees = np.degrees(angle)

        # 创建长方形补丁
        rect = patches.Rectangle(
            (rect_center_x - rectangle_height / 2, rect_center_y - rectangle_width / 2),
            rectangle_height, rectangle_width,
            angle=angle_degrees, color='white', alpha=0.5
        )

        # 添加长方形到绘图中
        plt.gca().add_patch(rect)

    # 设置图形参数
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')



if __name__ == '__main__':
    # 正弦曲线参数
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    # 绘制正弦曲线
    plt.plot(x, y, label='Sine Curve')
    plotrec(X,Y)
    plt.show()