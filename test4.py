import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend('Agg')

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 🔧 创建一个山峰一个低谷的地形
# 山峰在(-0.2, -0.3)，高度为2
peak_x, peak_y = -0.2, -0.3
peak_height = 2
peak_width = 0.8  # 控制山峰的宽度

# 山谷在(1.0, -2.5)，深度为1
valley_x, valley_y = 1.0, -2.5
valley_depth = 1
valley_radius = 3  # 🔧 山谷影响半径改为3

# 🔧 新增Robot 2位置
robot2_x, robot2_y = -0.3, -1.8

# 🔧 水平基础平面（高度为0）
base_plane = np.zeros_like(X)  # 完全水平的基础平面

# 山峰：使用高斯函数
peak = peak_height * np.exp(-((X - peak_x)**2 + (Y - peak_y)**2) / (2 * peak_width**2))

# 🔧 山谷：使用负高斯函数，调整参数使影响半径为3
valley_sigma = valley_radius / 3  # σ = 1，这样3σ = 3
valley = -valley_depth * np.exp(-((X - valley_x)**2 + (Y - valley_y)**2) / (2 * valley_sigma**2))

# 组合地形
Z = base_plane + peak + valley

# 🔧 修复坐标对应问题 - 将连续坐标映射到网格索引
def get_terrain_height_at_position(x_pos, y_pos, X, Y, Z):
    """根据连续坐标获取地形高度"""
    # 找到最接近的网格点
    x_idx = np.argmin(np.abs(X[0, :] - x_pos))
    y_idx = np.argmin(np.abs(Y[:, 0] - y_pos))
    return Z[y_idx, x_idx]

# 🔧 计算各点的实际地形高度
robot1_terrain_height = get_terrain_height_at_position(peak_x, peak_y, X, Y, Z)
workpiece_terrain_height = get_terrain_height_at_position(valley_x, valley_y, X, Y, Z)  
robot2_terrain_height = get_terrain_height_at_position(robot2_x, robot2_y, X, Y, Z)

# 只画一个3D图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 🔧 使用从红到蓝的地形颜色映射（低处红色，高处蓝色）
surface = ax.plot_surface(X, Y, Z, cmap='coolwarm_r', linewidth=0, antialiased=True, alpha=0.7)

# 🔧 标记点 - 使用正确的地形高度和鲜艳颜色
# Robot 1 - 深蓝色三角
ax.scatter([peak_x], [peak_y], [robot1_terrain_height + 0.3], color='blue', s=200, marker='^', 
           edgecolors='white', linewidth=3)
ax.text(peak_x + 0.3, peak_y, robot1_terrain_height + 0.5, 
        f'Robot 1\n({peak_x}, {peak_y})', 
        fontsize=10, color='blue', weight='bold')

# Workpiece - 黄色倒三角
ax.scatter([valley_x], [valley_y], [workpiece_terrain_height + 0.1], color='red', s=250, marker='v', 
           edgecolors='black', linewidth=3)
ax.text(valley_x + 0.3, valley_y, workpiece_terrain_height + 0.3, 
        f'Workpiece\n({valley_x}, {valley_y})', 
        fontsize=10, color='black', weight='bold')

# Robot 2 - 橙色方块
ax.scatter([robot2_x], [robot2_y], [robot2_terrain_height + 0.3], color='orange', s=220, marker='s', 
           edgecolors='black', linewidth=3)
ax.text(robot2_x + 0.3, robot2_y, robot2_terrain_height + 0.3, 
        f'Robot 2\n({robot2_x}, {robot2_y})', 
        fontsize=10, color='black', weight='bold')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')
ax.set_zlim(-1.2, 2.3)

plt.tight_layout()
plt.savefig('terrain_final.png', dpi=300, bbox_inches='tight')
print("📊 图片已保存为: terrain_final.png")

plt.close()
print("✅ 完成！只生成了一个3D地形图")