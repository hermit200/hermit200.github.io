
Pandas 是一个强大的 **数据分析和数据处理** 库，广泛应用于金融、统计、机器学习等领域。它提供了两种主要的数据结构：**Series** 和 **DataFrame**，让数据操作变得简洁高效。

---

### 1. 什么是 Pandas？
Pandas 是基于 NumPy 构建的，专门用于处理 **结构化数据**（如表格数据、时间序列数据等）的工具。  
它的特点包括：
- 强大的数据读取和写入功能（支持 CSV、Excel、SQL 等格式）。
- 简单易用的数据清洗和处理工具。
- 高效的分组和聚合操作。
- 适合数据分析和可视化的接口。

---

### 2. 核心数据结构
#### a. Series（一维数据）
类似于一维数组或 Python 的字典，可以包含索引和值。
```python
import pandas as pd

# 创建一个 Series
data = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("Series 数据:\n", data)

# 访问数据
print("通过索引访问:", data['b'])  # 输出 20
print("支持布尔筛选:", data[data > 20])  # 输出 c 和 d 的值
```

#### b. DataFrame（二维数据）
类似于电子表格或数据库表，由行和列组成。
```python
# 创建一个 DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Score": [90, 80, 85]
}
df = pd.DataFrame(data)
print("DataFrame 数据:\n", df)

# 访问数据
print("访问列:\n", df['Name'])
print("访问行:\n", df.loc[1])  # 通过标签访问
print("访问行:\n", df.iloc[2])  # 通过位置访问
```

---

### 3. 数据操作
#### a. 数据读取与写入
Pandas 支持多种文件格式的数据读取和保存。
```python
# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 写入 CSV 文件
df.to_csv('output.csv', index=False)

# 读取 Excel 文件
df_excel = pd.read_excel('data.xlsx')

# 写入 Excel 文件
df_excel.to_excel('output.xlsx', index=False)
```

#### b. 数据选择与筛选
```python
# 筛选满足条件的数据
filtered_df = df[df['Age'] > 30]
print("筛选结果:\n", filtered_df)

# 选择多列
subset = df[['Name', 'Score']]
print("列子集:\n", subset)

# 添加新列
df['Passed'] = df['Score'] > 85
print("添加新列:\n", df)
```

#### c. 缺失值处理
```python
# 创建一个包含缺失值的 DataFrame
data = {
    "Name": ["Alice", "Bob", None],
    "Age": [25, None, 35],
    "Score": [90, 80, None]
}
df = pd.DataFrame(data)

# 检测缺失值
print("是否缺失:\n", df.isnull())

# 填充缺失值
df_filled = df.fillna(0)  # 用 0 填充
print("填充后的数据:\n", df_filled)

# 删除缺失值
df_dropped = df.dropna()  # 删除含缺失值的行
print("删除缺失值后的数据:\n", df_dropped)
```

---

### 4. 数据聚合与分组
Pandas 提供了强大的分组和聚合功能，适用于数据统计和分析。
```python
# 示例数据
data = {
    "Department": ["HR", "IT", "HR", "IT", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Salary": [5000, 6000, 4500, 7000, 4800]
}
df = pd.DataFrame(data)

# 按部门分组并计算平均工资
grouped = df.groupby('Department')['Salary'].mean()
print("按部门计算平均工资:\n", grouped)

# 统计每个部门的员工数
employee_count = df.groupby('Department')['Employee'].count()
print("每个部门的员工数:\n", employee_count)
```

---

### 5. 时间序列分析
Pandas 提供了内置的时间序列处理工具。
```python
# 创建时间序列数据
date_range = pd.date_range(start='2023-01-01', periods=7, freq='D')
data = pd.Series([100, 200, 150, 300, 400, 350, 500], index=date_range)
print("时间序列数据:\n", data)

# 滑动窗口计算（移动平均）
rolling_mean = data.rolling(window=3).mean()
print("移动平均:\n", rolling_mean)

# 重新采样（按周计算总和）
resampled = data.resample('W').sum()
print("按周重新采样:\n", resampled)
```

---

### 6. 数据合并与连接
Pandas 支持类似 SQL 的表操作，包括连接、合并、拼接等。
```python
# 创建两个 DataFrame
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Score': [90, 85]})

# 合并（类似内连接）
merged = pd.merge(df1, df2, on='ID')
print("合并结果:\n", merged)

# 拼接（按行）
df3 = pd.DataFrame({'ID': [3], 'Name': ['Charlie']})
concatenated = pd.concat([df1, df3])
print("拼接结果:\n", concatenated)
```

---

### 7. 可视化
Pandas 支持简单的绘图操作。
```python
import matplotlib.pyplot as plt

# 示例数据
data = {"Age": [25, 30, 35, 40], "Score": [85, 90, 75, 80]}
df = pd.DataFrame(data)

# 绘制柱状图
df.plot(x='Age', y='Score', kind='bar', title='Age vs Score')
plt.show()
```

---

### 8. 数据清洗与预处理

#### a. 处理重复值
```python
# 示例数据
data = {
    "Name": ["Alice", "Bob", "Alice", "Eve"],
    "Age": [25, 30, 25, 40],
    "Score": [90, 85, 90, 80]
}
df = pd.DataFrame(data)

# 检测重复值
print("是否重复:\n", df.duplicated())

# 删除重复值
df_no_duplicates = df.drop_duplicates()
print("删除重复值后:\n", df_no_duplicates)
```

#### b. 数据类型转换
```python
# 示例数据
data = {"Age": ["25", "30", "N/A", "40"], "Score": ["85", "90", "80", "NaN"]}
df = pd.DataFrame(data)

# 替换和转换数据类型
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # 转为数字类型
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

print("数据类型转换后的 DataFrame:\n", df)
```

#### c. 字符串操作
Pandas 提供了丰富的字符串处理方法。
```python
# 示例数据
data = {"Name": ["  Alice ", "BOB", "eve "], "City": [" New York ", "LOS ANGELES", " seattle"]}
df = pd.DataFrame(data)

# 清理字符串
df['Name'] = df['Name'].str.strip().str.capitalize()
df['City'] = df['City'].str.strip().str.title()
print("清理后的数据:\n", df)
```

---

### 9. 高级聚合与分组分析

#### a. 多重分组与聚合
```python
# 示例数据
data = {
    "Department": ["HR", "HR", "IT", "IT", "IT"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Salary": [5000, 6000, 7000, 8000, 7500],
    "Bonus": [500, 600, 700, 800, 750]
}
df = pd.DataFrame(data)

# 多重分组
grouped = df.groupby(['Department', 'Employee']).sum()
print("多重分组结果:\n", grouped)

# 聚合函数
agg_result = df.groupby('Department').agg({
    "Salary": "mean",
    "Bonus": ["sum", "max"]
})
print("聚合分析:\n", agg_result)
```

#### b. 分组排名
```python
# 示例数据
data = {
    "Department": ["HR", "HR", "IT", "IT", "IT"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Salary": [5000, 6000, 7000, 8000, 7500]
}
df = pd.DataFrame(data)

# 分组排名
df['Rank'] = df.groupby('Department')['Salary'].rank(ascending=False)
print("分组排名:\n", df)
```


---

### 10. 数据合并与连接

#### a. 多表合并
```python
# 示例数据
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 3], 'Score': [90, 85]})

# 外连接
merged_outer = pd.merge(df1, df2, on='ID', how='outer')
print("外连接结果:\n", merged_outer)
```

#### b. 多索引操作
```python
# 设置多重索引
df = df.set_index(['Department', 'Employee'])
print("多索引 DataFrame:\n", df)

# 按索引访问
print("按索引访问:\n", df.loc[('IT', 'David')])
```

---

### 11. 综合案例

**数据描述**

假设我们有一份电子商务销售数据集，包含以下列：
- `OrderID`：订单编号
- `Product`：产品名称
- `Quantity`：数量
- `Price`：单价
- `OrderDate`：订单日期

**任务**
1. 找出每个产品的总销售额。
2. 计算每个月的销售趋势。
3. 找出销量最高的产品。

**实现代码**
```python
# 示例数据
data = {
    "OrderID": [1, 2, 3, 4, 5],
    "Product": ["A", "B", "A", "C", "B"],
    "Quantity": [2, 1, 5, 1, 3],
    "Price": [10, 20, 10, 30, 20],
    "OrderDate": ["2023-01-01", "2023-01-03", "2023-02-01", "2023-02-05", "2023-03-01"]
}
df = pd.DataFrame(data)

# 1. 添加销售额列
df['TotalSales'] = df['Quantity'] * df['Price']

# 2. 按产品分组计算总销售额
product_sales = df.groupby('Product')['TotalSales'].sum()
print("按产品总销售额:\n", product_sales)

# 3. 解析日期并按月统计销售趋势
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
monthly_sales = df.resample('M', on='OrderDate')['TotalSales'].sum()
print("每月销售趋势:\n", monthly_sales)

# 4. 找出销量最高的产品
top_product = df.groupby('Product')['Quantity'].sum().idxmax()
print("销量最高的产品:", top_product)
```

---
