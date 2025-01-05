
本文参考了周志华老师的《机器学习》（俗称“西瓜书”）。这里是 **第六章“支持向量机”** 的阅读笔记。本文归纳整理了核心知识点，并且记录了我的思考，希望对你有所帮助🎉

### **1. 支持向量机的基本概念**

 **1.1 什么是支持向量机？**
支持向量机是一种监督学习算法，可用于 **分类** 和 **回归** 问题。它通过构造一个或多个超平面，将不同类别的数据点尽可能正确地分开。

1. **超平面**：
   - 分类边界，用于将数据划分为不同类别。
   - 公式：

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$


2. **支持向量**：
   - 靠近分类边界的样本点，决定了分类超平面的位置。
   - 支持向量是训练模型时的重要点，其对分类结果影响最大。

3. **分类间隔（Margin）**：
   - 数据点到超平面的最小距离。
   - 支持向量机的目标是找到一个最大化分类间隔的超平面。

---

**2. 线性可分支持向量机**

 **2.1 基本思想**
在数据可以被线性分割的情况下，支持向量机的目标是找到一个能最大化分类间隔的超平面。

 **2.2 优化目标**
- 分类间隔公式：

$$
\gamma = \frac{2}{\|\mathbf{w}\|}
$$

- 最大化分类间隔等价于最小化：

$$
\frac{1}{2} \|\mathbf{w}\|^2
$$

同时满足以下约束：

$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
$$



---

### **3. 线性不可分支持向量机**

**3.1 问题背景**
现实数据中，通常无法线性分割。例如，两个类别可能互相交叉。这种情况下，我们需要允许一定的分类错误。

**3.2 引入松弛变量**
- **软间隔 SVM**：
  - 引入松弛变量 xi_i：

$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

**3.3 优化目标**
新的目标函数为：

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m \xi_i
$$

- C：正则化参数，用于权衡分类间隔与分类错误的影响。

---

### **4. 核方法**

**4.1 核函数的引入**
在数据无法线性分割时，我们可以通过核函数将数据映射到高维空间，在高维空间中实现线性分割。

**4.2 常见核函数**
1. **线性核**：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j
$$

2. **多项式核**：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d
$$

3. **高斯核（RBF 核）**：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$

4. **Sigmoid 核**：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\mathbf{x}_i^T \mathbf{x}_j + c)
$$

---

### **5. 支持向量回归（SVR）**

**5.1 epsilon-不敏感间隔**
SVR 中定义了一个 epsilon-不敏感间隔，即只关心预测值与真实值之间的误差是否超过 epsilon。

**5.2 优化目标**
目标函数为：

$$
\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m (\xi_i + \xi_i^*)
$$

约束条件：

$$
\begin{cases}
y_i - (\mathbf{w}^T \mathbf{x}_i + b) \leq \epsilon + \xi_i \\
(\mathbf{w}^T \mathbf{x}_i + b) - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases}
$$


---

### **6. Scikit-learn 实现支持向量机**

以下是 SVM 分类和回归的代码实现。

**6.1 SVM 分类（使用 RBF 核）**

```python
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# 1. 生成数据
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# 2. 创建 SVM 模型
model = SVC(kernel='rbf', C=1, gamma=0.5)
model.fit(X, y)

# 3. 可视化分类边界
def plot_svm_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title("SVM with RBF Kernel")
    plt.show()

plot_svm_decision_boundary(model, X, y)
```

**结果展示**

![](https://i.ibb.co/FXf7hfV/SVM.png)

**6.2 SVM 回归**

```python
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 生成数据
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# 2. 创建 SVR 模型
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X, y)

# 3. 预测
y_pred = svr_rbf.predict(X)

# 4. 可视化结果
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_pred, color='navy', lw=2, label='RBF model')
plt.title("SVR with RBF Kernel")
plt.legend()
plt.show()
```
**结果展示**

![](https://i.ibb.co/kK38WCv/SVR.png)
---

### **7. 头脑风暴**

1. **总结对比**

| **内容**              | **描述**                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| 支持向量机的目标      | 构造一个超平面以最大化分类间隔或实现非线性分割                           |
| 核方法的引入          | 通过核函数将数据映射到高维空间，解决线性不可分问题                      |
| SVM 分类与回归        | 同时适用于分类任务（SVC）和回归任务（SVR）                              |
| 参数 C, gamma    | C 控制间隔与误差的权衡，gamma 控制 RBF 核的影响范围            |

2. **理解线性不可分向量机的松弛变量**

![](https://i.ibb.co/c2BDTBG/image.png)

3.**支持向量机是否真的需要所有数据？为什么支持向量是关键？**

 **思考**：
SVM 的名称就来自 **支持向量**，这些点是离分类边界最近的样本。那么：
- 为什么支持向量足以决定分类边界？
- 其他样本对模型是否完全无用？

 **解析**：
- **核心原理**：
  SVM 的目标是最大化分类间隔，而分类边界只由支持向量决定。非支持向量（即远离边界的点）对边界的优化贡献为 0，因此可以舍弃。
- **实际意义**：
  - 在处理高维稀疏数据时，SVM 可以显著减少计算量。
  - 这也启发了核方法，支持向量成为计算核函数的重要子集。

 
---

4. **SVM 如何在高维数据中保持强大？是否会遭遇“维度灾难”？**

 **思考**：
SVM 通过核函数将数据映射到高维空间以实现线性可分，但高维数据通常会导致“维度灾难”（计算量指数增长）。那么：
- 为什么 SVM 在高维空间中依然表现良好？
- 核函数的计算复杂度如何避免“维度灾难”？

**解析**：
- **核函数的作用**：
  - 核函数通过“内积”隐式计算高维映射，避免显式构造高维特征向量。
  - 计算复杂度仅与样本数量和支持向量数量相关，而与高维空间的维度无关。
- **高维的优势**：
  - 高维空间中，数据更容易线性可分，因此 SVM 能够找到更优的超平面。
  - 高维特征可能带来过拟合风险，但正则化参数 \(C\) 和核方法帮助缓解。
---

5. **几个核函数的对比**


| **核函数**     | **优点**                                                                                     | **缺点**                                                                                     | **适用场景**                                                        |
|----------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| **线性核**     | - 计算简单，速度快。<br>- 对线性可分数据效果好。<br>- 不易过拟合。                             | - 无法处理非线性数据。<br>- 表现受限于特征间的线性关系。                                    | - 特征与类别之间线性关系强的场景，例如文本分类或高维稀疏数据。      |
| **多项式核**   | - 可处理一定程度的非线性关系。<br>- 参数（如阶数）可调，适配不同复杂度的数据分布。             | - 计算复杂度高，尤其是高阶多项式时。<br>- 容易过拟合（阶数高时）。                           | - 数据有显著非线性模式，但规律性较强的情况，例如形状识别。          |
| **高斯核（RBF 核）** | - 能处理复杂的非线性关系。<br>- 适合大多数数据分布，具有普适性。<br>- 参数 gamma 可调，灵活性强。 | - 参数选择敏感，gamma不当时可能过拟合或欠拟合。<br>- 难以解释映射后的高维空间含义。    | - 复杂非线性数据，例如图像分类、生物信息学数据分析。                |
| **Sigmoid 核** | - 类似于神经网络中的激活函数，可在一定程度上模拟神经网络。                                   | - 参数敏感，效果依赖于参数alpha 和 c的设置。<br>- 可能不符合核函数的 Mercer 条件。 | - 在需要尝试模仿神经网络特性时使用，应用较少。                      |

---
### 文章参考

- 《机器学习（西瓜书）》
- 部分LaTeX 公式借助了AI的帮助



