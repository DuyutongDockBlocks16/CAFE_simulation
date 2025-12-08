import matplotlib.pyplot as plt

# =D2+1
# =AVERAGE(F2:F101)
# =IF(F2>0,C2,0)
# =SUM(F2:F101)
# =SUM(G2:G101)
# =G102/F103
# =IF(B2="task_completed",1,0)
# =SUM(H2:H101)
# =H102/100


# 数据
rounds = [f"$\\pi_{{{i}}}$" for i in range(1, 6)]  # pi_1, pi_2, ..., pi_5
avg_progress = [9.5, 7.18, 10, 9.74, 10]
timestep_for_per_object = [181.9, 201.4, 158.0, 199.7, 192.6]

# 绘图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 平均进度曲线
color1 = "royalblue"
ax1.plot(rounds, avg_progress, marker="o", color=color1, linewidth=2, label="Average Progress")
ax1.set_xlabel("Policy", fontsize=16)
ax1.set_ylabel("Average Progress", fontsize=14, color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

# 调整横坐标标签字体大小
ax1.set_xticks(range(len(rounds)))
ax1.set_xticklabels(rounds, fontsize=16)

# 标注 Average Progress
for i, value in enumerate(avg_progress):
    ax1.text(i, value + 0.1, f"{value:.2f}", ha="center", fontsize=12, color=color1)

# 第二个 y 轴：Timesteps per Placed Object
ax2 = ax1.twinx()
color2 = "darkorange"
ax2.plot(rounds, timestep_for_per_object, marker="s", color=color2, linewidth=2, linestyle="--", label="Timesteps per Placed Object")
ax2.set_ylabel("Timesteps per Placed Object", fontsize=14, color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

# 标注 Timesteps per Placed Object
for i, value in enumerate(timestep_for_per_object):
    ax2.text(i, value - 5, f"{value:.2f}", ha="center", fontsize=12, color=color2)

# 标题和网格
plt.title("Average Progress and Timesteps per Placed Object per Policy", fontsize=18, fontweight="bold")
fig.tight_layout()
ax1.grid(True, alpha=0.3)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

# ax1.set_ylim(0, 3.5) 
# ax2.set_ylim(140, 320) 

# 保存图像
plt.savefig("ind_progress_timestep_per_object_per_policy.png", dpi=300, bbox_inches="tight")

plt.show()
