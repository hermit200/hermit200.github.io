
本文参考了周志华老师的《机器学习》（俗称“西瓜书”）。这里是 **第三章“线性模型”** 的阅读笔记。本文专注于**对数几率回归**这一块，并且记录了我的思考，希望对你有所帮助🎉

#### **1. 原理**
Logistic 回归是一种用于 **二分类问题** 的模型。其核心思想是将 **线性回归** 的输出通过一个非线性函数（Sigmoid 函数）映射到 \( [0, 1] \) 区间，解释为样本属于某个类别的概率。

---

#### **2.基本思想**

1. **分类概率**：
   Logistic 回归假设目标值 \(y \in \{0, 1\}\) 与输入特征 \(x\) 的关系可以用以下公式表示：

   $$
   P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}
   $$

   其中：
 - 线性模型的输出：
 
  $$
  w^T x + b
  $$

- 将线性模型的输出通过 **指数函数** 映射到非线性空间：


$$
e^{-(w^T x + b)}
$$



- **Sigmoid 函数**：将任意实数压缩到 \( [0, 1] \) 区间：



$$
  \sigma(z) = \frac{1}{1 + e^{-z}}
$$



2. **二分类决策**：
   - Logistic 回归的输出为分类概率：
   
$$
     \hat{y} = P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

   - 通过概率进行分类：
   
$$
     y = 
     \begin{cases} 
     1, & \text{若 } \hat{y} \geq 0.5 \\ 
     0, & \text{若 } \hat{y} < 0.5
     \end{cases}
$$

---

#### **3.目标函数**

1. **最大化似然函数**：
   Logistic 回归的目标是找到参数 \(w\) 和 \(b\)，使得预测概率与真实标签 \(y\) 的符合程度最大化。这通过 **最大化数据的似然函数（Log-Likelihood）** 实现：

$$
   L(w, b) = \prod_{i=1}^m P(y_i|x_i)
$$

   其中：
 其中：
- 每个样本的概率：

$$
  P(y_i|x_i) = \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}
$$

- 样本总数：

$$
  m
$$


2. **对数似然函数**：
   为方便计算，取对数得到对数似然函数：

$$
\ell(w, b) = \sum_{i=1}^m \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]
$$


3. **最小化对数损失函数（Log-Loss）**：
   为了简化优化目标，最小化损失函数的负对数似然形式：

$$
L(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]
$$

其中：
- 第一项：

$$
  y_i \log(\hat{y}_i)
$$

  若 y_i = 1，最大化预测为正类的概率。

- 第二项：

$$
  (1 - y_i) \log(1 - \hat{y}_i)
$$

  若 y_i = 0，最大化预测为负类的概率。


---

#### **4.适用场景**

1. **任务类型**：
   Logistic 回归适用于 **二分类任务**，例如：
   - 健康诊断（是否患病）。
   - 垃圾邮件分类。
   - 客户流失预测。

2. **数据特性**：
   - Logistic 回归假设特征与目标值之间存在 **线性可分** 的关系（通过超平面划分两类）。
   - 对于非线性数据，需结合特征工程（如多项式特征）或使用更复杂的模型（如 SVM 或神经网络）。

3. **优点**：
   - 计算效率高。
   - 输出概率值，易于解释。
   - 适用于中小规模数据集。

4. **缺点**：
   - 对线性不可分数据表现较差。
   - 对离群点敏感。

---

### 5.头脑风暴

1. **Sigmoid 函数和Logistic回归**：
![](https://i.ibb.co/rkLQ1s8/image.png)

2. **损失函数解释**
![](https://i.ibb.co/yFNhP8f/image.png)

**对数损失函数是一个凸函数**（对线性模型而言），这意味着可以通过梯度下降找到全局最优解


3. **代码实现**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 1. 生成模拟数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 创建并训练 Logistic Regression 模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. 计算损失
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 预测类别概率
loss = log_loss(y_test, y_pred_proba)  # 使用 Scikit-learn 自带的 log_loss 计算
print(f"模型的对数损失值：{loss:.4f}")

# 4. 可视化决策边界
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

plot_decision_boundary(model, X_test, y_test)
```
**结果展示**
![](https://i.ibb.co/0nnzXDL/image.png)


### 文章参考

- 《机器学习（西瓜书）》
- 部分LaTeX 公式借助了AI的帮助