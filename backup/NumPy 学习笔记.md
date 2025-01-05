
#### 1. 什么是 NumPy？
NumPy 是一个用于 **科学计算** 和 **数据分析** 的基础库。它提供了高效的多维数组操作，并包含大量数学函数，如线性代数、傅里叶变换和随机数生成。

**特点**：
- 提供强大的 `ndarray` 对象（多维数组）。
- 高效的计算性能（底层由 C 实现）。
- 支持广播机制，简化数组运算。
- 丰富的数学和统计函数库。

#### 2. 核心功能概览
##### a. 创建数组
```python
import numpy as np

# 创建一维数组
arr1 = np.array([1, 2, 3, 4])

# 创建二维数组
arr2 = np.array([[1, 2], [3, 4]])

# 创建特定形状的数组
zeros = np.zeros((2, 3))  # 全零数组
ones = np.ones((3, 2))    # 全一数组
identity = np.eye(3)      # 单位矩阵
random = np.random.rand(2, 3)  # 随机数数组

print("一维数组:", arr1)
print("二维数组:\n", arr2)
```

##### b. 数组基本操作
```python
# 数组属性
print(arr2.shape)  # (2, 2)，形状
print(arr2.dtype)  # int32，数据类型
print(arr2.ndim)   # 维度数量

# 数组运算
arr3 = np.array([10, 20, 30, 40])
result = arr1 + arr3  # 元素加法
print("元素加法:", result)

# 广播机制
arr4 = np.array([[1], [2], [3]])
broadcast_result = arr4 + arr1
print("广播结果:\n", broadcast_result)
```

##### c. 索引和切片
```python
# 一维数组索引
print(arr1[0])   # 获取第一个元素
print(arr1[-1])  # 获取最后一个元素

# 二维数组索引
print(arr2[0, 1])  # 第一行第二列
print(arr2[:, 0])  # 获取第一列

# 切片
print(arr1[1:3])    # 获取索引1到3的元素
print(arr2[1:, :])  # 获取从第二行开始的所有行
```

#### 3. 高级功能
##### a. 线性代数
```python
from numpy.linalg import inv, det

matrix = np.array([[1, 2], [3, 4]])
inverse = inv(matrix)  # 计算逆矩阵
determinant = det(matrix)  # 计算行列式

print("矩阵:\n", matrix)
print("逆矩阵:\n", inverse)
print("行列式:", determinant)
```

##### b. 随机数生成
```python
# 固定种子
np.random.seed(42)

# 生成随机数
rand_array = np.random.rand(3, 3)  # 均匀分布
rand_int = np.random.randint(0, 10, size=(2, 3))  # 随机整数

print("随机数数组:\n", rand_array)
print("随机整数数组:\n", rand_int)
```

##### c. 数组重塑和拼接
```python
# 重塑
arr5 = np.arange(12).reshape(3, 4)
print("重塑后的数组:\n", arr5)

# 拼接
arr6 = np.array([[1, 2], [3, 4]])
arr7 = np.array([[5, 6]])
concatenated = np.concatenate((arr6, arr7), axis=0)  # 按行拼接
print("拼接结果:\n", concatenated)
```

#### 4. 简单例子
**数据归一化**
```python
data = np.array([5, 10, 15, 20])
normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
print("归一化结果:", normalized)
```

**求解方程组**
```python
# 线性方程组: Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print("方程组解:", x)
```


#### 5. 数组操作进阶
##### a. 条件筛选
利用布尔索引可以高效筛选数组中的元素。
```python
arr = np.array([10, 15, 20, 25, 30])

# 筛选大于20的元素
filtered = arr[arr > 20]
print("大于20的元素:", filtered)

# 替换满足条件的元素
arr[arr > 20] = 99
print("替换后的数组:", arr)
```

##### b. 数学与统计函数
NumPy 提供了大量用于数学和统计计算的函数。
```python
data = np.array([[1, 2, 3], [4, 5, 6]])

# 数学计算
print("最大值:", np.max(data))
print("最小值:", np.min(data))
print("平均值:", np.mean(data))
print("标准差:", np.std(data))
print("按列求和:", np.sum(data, axis=0))

# 求累计和
cumsum = np.cumsum(data)
print("累计和:", cumsum)
```

##### c. 矩阵操作
```python
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# 矩阵乘法
dot_product = np.dot(matrix_a, matrix_b)
print("矩阵乘法结果:\n", dot_product)

# 元素逐一乘法（Hadamard积）
elementwise_product = matrix_a * matrix_b
print("逐元素乘积:\n", elementwise_product)
```

---

#### 6. 实用场景案例
##### a. 图像数据处理
图像可以用 NumPy 数组表示（通常是三维数组），对其操作可以实现快速处理。
```python
# 假设有一个模拟的灰度图像
image = np.random.randint(0, 256, (5, 5))  # 5x5随机像素值
print("原始图像:\n", image)

# 图像归一化到[0, 1]
normalized_image = image / 255.0
print("归一化图像:\n", normalized_image)

# 图像二值化
binary_image = (image > 128).astype(int)
print("二值化图像:\n", binary_image)
```

##### b. 时间序列数据生成
在科学实验或金融数据处理中，经常需要生成连续的时间序列数据。
```python
# 生成等间距时间序列
time_series = np.linspace(0, 10, 50)  # 从0到10等间距生成50个数
print("时间序列:", time_series)

# 对时间序列应用正弦函数
sin_wave = np.sin(time_series)
print("正弦波:", sin_wave)
```

##### c. 快速傅里叶变换（FFT）
NumPy 的 `fft` 模块支持信号处理中的快速傅里叶变换。
```python
from numpy.fft import fft

# 生成模拟信号
signal = np.sin(2 * np.pi * np.arange(50) / 10)

# 进行FFT
fft_result = fft(signal)
print("FFT结果:\n", fft_result)
```

---

#### 7. 性能优化
##### a. 向量化运算
NumPy 的向量化操作极大提高了计算效率，避免使用慢速的 Python 循环。
```python
# 普通循环计算平方
data = np.arange(1, 10001)
squares_loop = [x**2 for x in data]

# 使用NumPy向量化
squares_vectorized = data**2
print("结果是否一致:", np.array_equal(squares_loop, squares_vectorized))
```

##### b. 使用内存视图（切片）
切片是数组的一部分，使用的是相同的内存空间，因此无需复制数据。
```python
arr = np.arange(10)
slice_view = arr[2:5]  # 切片创建视图
slice_view[0] = 99
print("原数组受影响:", arr)
```

---
#### 本文参考
- 菜鸟教程：https://www.runoob.com/numpy/numpy-tutorial.html