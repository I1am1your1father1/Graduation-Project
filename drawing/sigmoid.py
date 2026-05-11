import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 1000)
y = sigmoid(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=3, label='Sigmoid(x)')

plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
plt.axvline(x=0, color='g', linestyle='--', linewidth=1.5, alpha=0.8)

plt.title('Sigmoid Function', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('σ(x)', fontsize=12)
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig('/home/dingning/毕设/Mine/drawing/sigmoid.png', dpi=300)
print("✅ Sigmoid 图像已保存！")