import matplotlib.pyplot as plt

# 数据
rounds = [f"$\\pi_{{{i}}}$" for i in range(1, 6)]
timesteps_per_object = [181.9, 201.4, 158, 199.7, 192.6]

# 绘图
plt.figure(figsize=(12, 7))
plt.plot(rounds, timesteps_per_object, marker='o', color='royalblue', linewidth=3)

# 数据标签
for i, value in enumerate(timesteps_per_object):
    plt.text(i, value + 1, f"{value:.2f}", ha='center', fontsize=14)

# 标签和标题
plt.xlabel("Policy", fontsize=20)
plt.ylabel("Timesteps per Placed Object", fontsize=20)
plt.title("Timesteps per Placed Object per Policy", fontsize=20, fontweight='bold')

plt.xticks(fontsize=26)
plt.yticks(fontsize=20)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图像
plt.savefig("ind_timesteps_per_object.png", dpi=300, bbox_inches="tight")

plt.show()
