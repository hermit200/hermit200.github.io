# 文档使用说明

本文档主要是我个人在学习python基础时的学习笔记，主要涉及Python 的核心概念与常用功能，并提供一定的简单实例，不涉及过于复杂的高级用法。


##  一、数据类型概述

  在 Python 中，数据类型用于定义变量存储的数据种类。Python 是一种动态类型语言，变量的数据类型会根据赋值自动推断。以下是 Python 常见的数据类型：

  ### 1. 数字类型

  数字类型用于存储数值，主要包括以下三种类型：

  - **整数（int）**: 表示没有小数部分的数字。
  - **浮点数（float）**: 表示有小数部分的数字。
  - **复数（complex）**: 包含实部和虚部的数字。

  **示例代码**：

  ```python
  # 整数
  x = 10
  print(type(x))  # 输出: <class 'int'>

  # 浮点数
  y = 3.14
  print(type(y))  # 输出: <class 'float'>

  # 复数
  z = 2 + 3j
  print(type(z))  # 输出: <class 'complex'>

  ```
  ### 2. 字符串（str）

  字符串用于表示一系列字符。字符串可以用单引号、双引号或三引号定义。

  **示例代码**：

  ```python
  # 单引号
  name = 'Alice'

  # 双引号
  greeting = "Hello, World!"

  # 三引号（可用于多行字符串）
  message = '''This is
  a multiline
  string.'''

  print(type(name))  # 输出: <class 'str'>
  ```

  ### 3. 布尔值（bool）

  布尔值表示逻辑值，只有两个可能的取值：`True` 和 `False`。

  **示例代码**：

  ```python
  is_active = True
  is_logged_in = False
  print(type(is_active))  # 输出: <class 'bool'>
  ```

  ### 4. 容器类型

  容器类型用于存储多个值。常见的容器类型包括：

  - **列表（list）**: 有序且可变。
  - **元组（tuple）**: 有序但不可变。
  - **字典（dict）**: 键值对形式存储。
  - **集合（set）**: 无序且唯一。

  **示例代码**：

  ```python
  # 列表
  fruits = ['apple', 'banana', 'cherry']

  # 元组
  coordinates = (10, 20)

  # 字典
  person = {'name': 'Alice', 'age': 25}

  # 集合
  unique_numbers = {1, 2, 3, 3}  # 重复值会被去除

  print(type(fruits))  # 输出: <class 'list'>
  print(type(coordinates))  # 输出: <class 'tuple'>
  print(type(person))  # 输出: <class 'dict'>
  print(type(unique_numbers))  # 输出: <class 'set'>
  ```

  ---


## 二、输入与输出

  在 Python 中，动态输入和格式化输出是编写交互式程序的基础。`input` 函数用于获取用户输入，而 `f-string` 是一种高效的字符串格式化方法。

  ### 1. 使用 `input` 获取用户输入

  `input` 是一个内置函数，用于接收用户从键盘输入的数据。输入的内容总是以字符串类型返回。常见用法如下：

  ```python
  # 获取用户输入
  name = input("请输入你的名字：")
  age = input("请输入你的年龄：")

  # 注意：所有输入都以字符串形式返回
  print(f"你的名字是 {name}，你的年龄是 {age}。")
  ```

  **转换输入类型**
  如果需要将输入转换为其他数据类型（如整数或浮点数），可以使用相应的类型转换函数，例如 `int()` 或 `float()`。

  ```python
  # 获取并转换年龄为整数
  age = int(input("请输入你的年龄："))

  # 进行简单的逻辑判断
  if age >= 18:
      print("你已经成年了！")
  else:
      print("你还是未成年人！")
  ```

  ### 2. 使用 `f-string` 格式化输出

  `f-string` 是一种快速简洁的字符串格式化方法，从 Python 3.6 开始支持。通过在字符串前加上 `f`，可以在字符串中嵌入变量或表达式。

  **基本语法**：

  ```python
  name = "Alice"
  age = 25

  # 使用 f-string
  print(f"我的名字是 {name}，我今年 {age} 岁。")
  ```

  **嵌入表达式**
  `f-string` 支持在大括号 `{}` 中嵌入任意表达式，例如计算、方法调用等。

  ```python
  # 嵌入表达式
  number = 5
  print(f"5 的平方是 {number ** 2}。")
  ```

  **控制输出格式**
  `f-string` 支持控制浮点数的精度、对齐方式等：

  ```python
  # 浮点数保留两位小数
  pi = 3.14159
  print(f"圆周率是 {pi:.2f}")  # 输出: 圆周率是 3.14

  # 对齐和宽度设置
  name = "Alice"
  print(f"|{name:<10}|")  # 左对齐，宽度为10
  print(f"|{name:>10}|")  # 右对齐，宽度为10
  ```
  ---


## 三、列表：Python 中的万能容器

  列表是 Python 中最常用的数据结构之一，它是一种有序、可变的容器，可以存储任意类型的元素。

  ### 1. 创建列表

  列表使用方括号 `[]` 定义，元素之间用逗号 `,` 分隔。

  ```python
  # 空列表
  empty_list = []

  # 包含整数的列表
  numbers = [1, 2, 3, 4, 5]

  # 包含不同数据类型的列表
  mixed_list = [1, "apple", 3.14, True]

  # 嵌套列表
  nested_list = [[1, 2], [3, 4]]
  ```

  ### 2. 访问列表元素

  可以通过索引访问列表元素，索引从 `0` 开始，也可以使用负数索引从末尾开始。

  ```python
  fruits = ["apple", "banana", "cherry"]

  # 访问第一个元素
  print(fruits[0])  # 输出: apple

  # 访问最后一个元素
  print(fruits[-1])  # 输出: cherry
  ```

  ### 3.  修改列表元素

  列表是可变的，因此可以直接通过索引修改元素的值。

  ```python
  fruits = ["apple", "banana", "cherry"]
  fruits[1] = "blueberry"
  print(fruits)  # 输出: ['apple', 'blueberry', 'cherry']
  ```

  ### 4. 添加元素

  **使用 `append()`**

  在列表末尾添加一个元素。

  ```python
  fruits.append("date")
  print(fruits)  # 输出: ['apple', 'banana', 'cherry', 'date']
  ```

  **使用 `insert()`**

  在指定位置插入一个元素。

  ```python
  fruits.insert(1, "orange")
  print(fruits)  # 输出: ['apple', 'orange', 'banana', 'cherry']
  ```

  ### 5. 删除元素

  **使用 `remove()`**

  根据值删除第一个匹配的元素。

  ```python
  fruits.remove("banana")
  print(fruits)  # 输出: ['apple', 'cherry']
  ```

  **使用 `pop()`**

  根据索引删除元素，默认删除最后一个。

  ```python
  fruits.pop()  # 删除最后一个元素
  print(fruits)  # 输出: ['apple', 'banana']
  ```

  **使用 `del`**

  使用索引或切片删除元素。

  ```python
  del fruits[0]  # 删除第一个元素
  print(fruits)  # 输出: ['banana', 'cherry']
  ```

  ### 6. 列表切片

  列表支持切片操作，用于获取子列表。

  ```python
  numbers = [0, 1, 2, 3, 4, 5, 6]

  # 获取索引 1 到 4 的子列表（不包括 5）
  print(numbers[1:5])  # 输出: [1, 2, 3, 4]

  # 从头到索引 3
  print(numbers[:4])  # 输出: [0, 1, 2, 3]

  # 从索引 2 到末尾
  print(numbers[2:])  # 输出: [2, 3, 4, 5, 6]

  # 步长切片
  print(numbers[::2])  # 输出: [0, 2, 4, 6]
  ```

  ### 7. 遍历列表

  **使用 `for` 循环**

  ```python
  fruits = ["apple", "banana", "cherry"]
  for fruit in fruits:
      print(f"I like {fruit}")
  ```

  **使用 `enumerate()`**

  获取索引和值。

  ```python
  for index, fruit in enumerate(fruits):
      print(f"Index {index}: {fruit}")
  ```

  ### 8. 常用列表方法总结

  | 方法           | 描述                                 |
  |----------------|--------------------------------------|
  | `append()`     | 在列表末尾添加元素                   |
  | `insert()`     | 在指定位置插入元素                   |
  | `remove()`     | 删除第一个匹配的值                   |
  | `pop()`        | 删除并返回指定位置的元素（默认最后） |
  | `index()`      | 返回指定值的索引                     |
  | `count()`      | 返回指定值在列表中的出现次数         |
  | `sort()`       | 对列表进行排序                       |
  | `reverse()`    | 反转列表                             |
  | `clear()`      | 清空列表                             |

  ### 9. 列表推导式

  列表推导式是一种简洁创建列表的方式。

  ```python
  # 创建一个平方列表
  squares = [x**2 for x in range(10)]
  print(squares)  # 输出: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

  # 筛选偶数
  evens = [x for x in range(10) if x % 2 == 0]
  print(evens)  # 输出: [0, 2, 4, 6, 8]
  ```
  ---

## 四、元组：不可变的有序集合

  元组是 Python 中的另一种序列类型，与列表类似，但最大的区别是 **元组是不可变的**，一旦创建，元素不能被修改。元组通常用于表示不需要更改的数据。

  ### 1. 创建元组

  元组使用小括号 `()` 定义，元素之间用逗号 `,` 分隔。如果只有一个元素，需要在末尾加一个逗号以区别单独的值。

  ```python
  # 空元组
  empty_tuple = ()

  # 含有多个元素的元组
  numbers = (1, 2, 3, 4, 5)

  # 含有一个元素的元组
  single_element = (42,)  # 注意逗号
  print(type(single_element))  # 输出: <class 'tuple'>

  # 不加逗号，会被视为一个普通值
  not_a_tuple = (42)
  print(type(not_a_tuple))  # 输出: <class 'int'>
  ```

  ### 2. 访问元组元素

  元组的元素可以通过索引访问，索引规则与列表相同，从 `0` 开始，也支持负数索引。

  ```python
  fruits = ("apple", "banana", "cherry")

  # 访问第一个元素
  print(fruits[0])  # 输出: apple

  # 访问最后一个元素
  print(fruits[-1])  # 输出: cherry
  ```

  ### 3. 元组的不可变性

  元组的不可变性意味着我们不能修改、添加或删除元组中的元素，但可以通过重新赋值来替换整个元组。

  ```python
  fruits = ("apple", "banana", "cherry")

  # 尝试修改元组中的元素会报错
  # fruits[1] = "orange"  # 会抛出 TypeError

  # 重新赋值
  fruits = ("orange", "grape")
  print(fruits)  # 输出: ('orange', 'grape')
  ```

  ### 4. 元组的常用操作

  元组支持一些常见的操作，如拼接、重复和解包。

  **元组拼接**

  ```python
  tuple1 = (1, 2, 3)
  tuple2 = (4, 5, 6)

  result = tuple1 + tuple2
  print(result)  # 输出: (1, 2, 3, 4, 5, 6)
  ```

  **元组重复**

  ```python
  numbers = (1, 2, 3)
  result = numbers * 2
  print(result)  # 输出: (1, 2, 3, 1, 2, 3)
  ```

  **元组解包**

  元组支持解包，将元组中的元素赋值给多个变量。

  ```python
  coordinates = (10, 20, 30)
  x, y, z = coordinates
  print(x, y, z)  # 输出: 10 20 30
  ```

  **嵌套元组解包**

  ```python
  nested_tuple = (1, (2, 3))
  a, (b, c) = nested_tuple
  print(a, b, c)  # 输出: 1 2 3
  ```

  ### 5. 元组的方法

  由于元组是不可变的，其方法非常有限，只有两个常用方法：

  | 方法         | 描述                                     |
  |--------------|------------------------------------------|
  | `count(x)`   | 返回元组中指定值 `x` 的出现次数          |
  | `index(x)`   | 返回元组中首次出现指定值 `x` 的索引位置  |

  **示例**：

  ```python
  numbers = (1, 2, 3, 2, 2, 4)

  # 统计值为 2 的出现次数
  print(numbers.count(2))  # 输出: 3

  # 查找值为 3 的索引
  print(numbers.index(3))  # 输出: 2
  ```

  ### 6. 元组与列表的转换

  尽管元组是不可变的，但可以通过类型转换在元组和列表之间切换。

  ```python
  # 元组转列表
  numbers = (1, 2, 3)
  numbers_list = list(numbers)
  print(numbers_list)  # 输出: [1, 2, 3]

  # 列表转元组
  numbers_list = [4, 5, 6]
  numbers_tuple = tuple(numbers_list)
  print(numbers_tuple)  # 输出: (4, 5, 6)
  ```

  ### 7. 元组的应用场景

  - **数据不可变的场景**: 如配置文件中的常量。
  - **作为字典的键**: 元组是不可变的，因此可以作为字典的键，而列表不可以。

  ```python
  # 元组作为字典的键
  locations = {
      (40.7128, -74.0060): "New York",
      (34.0522, -118.2437): "Los Angeles"
  }
  print(locations[(40.7128, -74.0060)])  # 输出: New York
  ```

  ---

## 五、字典：键值对的灵活存储

  字典是 Python 中的一种 **键值对（key-value pair）** 数据结构，它是无序的（Python 3.7 之后的实现是有序的）且可变的。字典非常适合存储和快速查找数据。

  ---

  ### 1. 创建字典

  字典使用大括号 `{}` 定义，键和值通过冒号 `:` 分隔，键值对之间用逗号 `,` 分隔。

  ```python
  # 空字典
  empty_dict = {}

  # 含有多个键值对的字典
  person = {
      "name": "Alice",
      "age": 25,
      "city": "New York"
  }
  ```

  字典中的键必须是不可变类型（如字符串、数字、元组），值可以是任意类型。

  ```python
  # 有效键：数字、字符串、元组
  valid_dict = {
      1: "one",
      "two": 2,
      (3, 4): "tuple_key"
  }
  ```

  ---

  ### 2. 访问字典元素

  通过键访问对应的值。如果键不存在，会抛出 `KeyError`。

  ```python
  person = {"name": "Alice", "age": 25}

  # 访问键对应的值
  print(person["name"])  # 输出: Alice

  # 使用 `get()` 方法访问
  print(person.get("age"))  # 输出: 25
  print(person.get("city", "Not Found"))  # 如果键不存在，返回默认值: Not Found
  ```

  ---

  ### 3. 修改字典

  **添加或更新键值对**

  直接使用键赋值即可。

  ```python
  person = {"name": "Alice", "age": 25}

  # 添加新键值对
  person["city"] = "New York"
  print(person)  # 输出: {'name': 'Alice', 'age': 25, 'city': 'New York'}

  # 更新已有键值对
  person["age"] = 30
  print(person)  # 输出: {'name': 'Alice', 'age': 30, 'city': 'New York'}
  ```

  **删除键值对**

  ```python
  # 使用 `pop()` 删除并返回指定键的值
  removed_value = person.pop("age")
  print(removed_value)  # 输出: 30
  print(person)  # 输出: {'name': 'Alice', 'city': 'New York'}

  # 使用 `del` 删除指定键
  del person["city"]
  print(person)  # 输出: {'name': 'Alice'}

  # 清空字典
  person.clear()
  print(person)  # 输出: {}
  ```

  ---

  ### 4. 遍历字典

  **遍历键**

  ```python
  person = {"name": "Alice", "age": 25, "city": "New York"}

  for key in person:
      print(key)  # 输出: name, age, city
  ```

  **遍历值**

  ```python
  for value in person.values():
      print(value)  # 输出: Alice, 25, New York
  ```

  **遍历键值对**

  ```python
  for key, value in person.items():
      print(f"{key}: {value}")
  # 输出:
  # name: Alice
  # age: 25
  # city: New York
  ```

  ---

  ### 5. 字典的常用方法

  | 方法             | 描述                                    |
  |------------------|-----------------------------------------|
  | `get(key)`       | 返回指定键的值，如果键不存在则返回 `None` 或默认值 |
  | `keys()`         | 返回所有键的视图对象                   |
  | `values()`       | 返回所有值的视图对象                   |
  | `items()`        | 返回所有键值对的视图对象               |
  | `update(dict2)`  | 用另一个字典的键值对更新当前字典       |
  | `pop(key)`       | 删除指定键，并返回对应的值             |
  | `clear()`        | 清空字典                               |

  **示例**：

  ```python
  person = {"name": "Alice", "age": 25}

  # 获取所有键
  print(person.keys())  # 输出: dict_keys(['name', 'age'])

  # 获取所有值
  print(person.values())  # 输出: dict_values(['Alice', 25])

  # 获取所有键值对
  print(person.items())  # 输出: dict_items([('name', 'Alice'), ('age', 25)])

  # 更新字典
  person.update({"city": "New York", "age": 30})
  print(person)  # 输出: {'name': 'Alice', 'age': 30, 'city': 'New York'}
  ```

  ---

  ### 6. 字典推导式

  字典推导式用于快速生成字典，语法与列表推导式类似。

  ```python
  # 创建键值对为数字及其平方的字典
  squares = {x: x**2 for x in range(5)}
  print(squares)  # 输出: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

  # 筛选字典
  original = {"a": 1, "b": 2, "c": 3}
  filtered = {k: v for k, v in original.items() if v > 1}
  print(filtered)  # 输出: {'b': 2, 'c': 3}
  ```

  ---

  ### 7. 字典的应用场景

  - **存储结构化数据**：如用户信息、配置文件等。
  - **实现简单映射**：如从键到值的快速查找。
  - **计数和统计**：结合 `collections.Counter`。

  **示例：统计字符出现的次数**：

  ```python
  text = "hello world"
  char_count = {}

  for char in text:
      char_count[char] = char_count.get(char, 0) + 1

  print(char_count)
  # 输出: {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
  ```

---
## 六、集合：处理无序且唯一的数据

  集合（`set`）是 Python 中的一种数据结构，用于存储无序且不重复的元素。它特别适合用于去重、集合运算（交集、并集等）以及快速查找。

  ---

  ### 1. 创建集合

  集合使用大括号 `{}` 或 `set()` 函数创建。需要注意，空集合只能用 `set()` 创建，否则会被认为是一个空字典。

  ```python
  # 空集合
  empty_set = set()

  # 包含多个元素的集合
  fruits = {"apple", "banana", "cherry"}

  # 自动去重
  numbers = {1, 2, 2, 3, 4}
  print(numbers)  # 输出: {1, 2, 3, 4}
  ```

  集合中的元素必须是不可变的（如数字、字符串、元组），但集合本身是可变的。

  ---

  ### 2. 访问集合元素

  集合是无序的，因此不支持索引访问或切片操作。

  ```python
  fruits = {"apple", "banana", "cherry"}

  # 检查元素是否存在
  print("apple" in fruits)  # 输出: True
  print("orange" in fruits)  # 输出: False
  ```

  ---

  ### 3. 修改集合

  **添加元素**

  使用 `add()` 方法向集合中添加一个元素。

  ```python
  fruits = {"apple", "banana"}
  fruits.add("cherry")
  print(fruits)  # 输出: {'apple', 'banana', 'cherry'}
  ```

  **删除元素**

  - 使用 `remove()` 删除指定元素，如果元素不存在，会抛出 `KeyError`。
  - 使用 `discard()` 删除指定元素，如果元素不存在，不会报错。
  - 使用 `pop()` 删除并返回集合中的一个随机元素。
  - 使用 `clear()` 清空集合。

  ```python
  fruits = {"apple", "banana", "cherry"}

  # 删除指定元素
  fruits.remove("banana")
  print(fruits)  # 输出: {'apple', 'cherry'}

  # 删除不存在的元素（不会报错）
  fruits.discard("orange")

  # 删除随机元素
  random_element = fruits.pop()
  print(random_element)  # 输出: cherry（具体值随机）
  print(fruits)  # 输出: {'apple'}

  # 清空集合
  fruits.clear()
  print(fruits)  # 输出: set()
  ```

  ---

  ### 4. 集合运算

  集合支持数学意义上的集合运算，如交集、并集、差集等。

  **并集（`|` 或 `union()`)**

  返回两个集合的所有元素（去重）。

  ```python
  set1 = {1, 2, 3}
  set2 = {3, 4, 5}
  print(set1 | set2)  # 输出: {1, 2, 3, 4, 5}
  print(set1.union(set2))  # 输出: {1, 2, 3, 4, 5}
  ```

  **交集（`&` 或 `intersection()`）**

  返回两个集合的公共元素。

  ```python
  print(set1 & set2)  # 输出: {3}
  print(set1.intersection(set2))  # 输出: {3}
  ```

  **差集（`-` 或 `difference()`）**

  返回只在第一个集合中存在的元素。

  ```python
  print(set1 - set2)  # 输出: {1, 2}
  print(set1.difference(set2))  # 输出: {1, 2}
  ```

  **对称差集（`^` 或 `symmetric_difference()`）**

  返回只在一个集合中存在的元素。

  ```python
  print(set1 ^ set2)  # 输出: {1, 2, 4, 5}
  print(set1.symmetric_difference(set2))  # 输出: {1, 2, 4, 5}
  ```

  ---

  ### 5. 集合的常用方法

  | 方法                        | 描述                                    |
  |-----------------------------|-----------------------------------------|
  | `add(x)`                    | 向集合中添加元素                       |
  | `remove(x)`                 | 删除集合中的指定元素，不存在则报错     |
  | `discard(x)`                | 删除集合中的指定元素，不存在也不报错   |
  | `pop()`                     | 删除并返回集合中的一个随机元素         |
  | `clear()`                   | 清空集合                               |
  | `union(other_set)`          | 返回两个集合的并集                     |
  | `intersection(other_set)`   | 返回两个集合的交集                     |
  | `difference(other_set)`     | 返回两个集合的差集                     |
  | `symmetric_difference(other_set)` | 返回两个集合的对称差集           |
  | `update(other_set)`         | 将其他集合的元素添加到当前集合         |

  ---

  ### 6. 集合推导式

  与列表推导式类似，集合推导式用于快速生成集合。

  ```python
  # 创建一个平方集合
  squares = {x**2 for x in range(10)}
  print(squares)  # 输出: {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

  # 筛选元素
  filtered = {x for x in range(10) if x % 2 == 0}
  print(filtered)  # 输出: {0, 2, 4, 6, 8}
  ```

  ---

  ### 7. 集合的应用场景

  - **数据去重**：集合自动去重功能非常方便。
  - **快速查找**：集合的查找复杂度为 O(1)。
  - **集合运算**：适用于需要处理交集、并集等数学运算的场景。

  **示例：数据去重**

  ```python
  numbers = [1, 2, 2, 3, 4, 4, 5]
  unique_numbers = set(numbers)
  print(unique_numbers)  # 输出: {1, 2, 3, 4, 5}
  ```

  ---
## 七、流程控制

  流程控制是 Python 中的重要组成部分，用于控制代码的执行顺序。主要包括条件语句、循环语句，以及控制循环的特殊语句。

  ---

  ### 1. 条件语句

  条件语句用于根据条件的真假执行不同的代码块，主要包括 `if`、`elif` 和 `else`。

  **基本语法**

  ```python
  age = int(input("请输入你的年龄："))

  if age < 18:
      print("你是未成年人。")
  elif age == 18:
      print("恭喜你刚刚成年！")
  else:
      print("你是成年人。")
  ```

  **嵌套条件语句**

  条件语句可以嵌套，但建议尽量避免过深的嵌套以保持代码清晰。

  ```python
  score = int(input("请输入你的考试成绩："))

  if score >= 60:
      if score >= 90:
          print("优秀！")
      else:
          print("及格，但还有提升空间。")
  else:
      print("不及格，请继续努力！")
  ```

  **单行条件语句**

  当条件和操作都很简单时，可以使用单行语法。

  ```python
  is_adult = True if age >= 18 else False
  print(f"是否成年: {is_adult}")
  ```

  ---

  ### 2. 循环语句

  Python 提供了两种主要的循环：`for` 和 `while`。

  #### 2.1 `for` 循环

  `for` 循环通常用于遍历序列（如列表、字符串等）或迭代器。

  ```python
  # 遍历列表
  fruits = ["apple", "banana", "cherry"]
  for fruit in fruits:
      print(f"I like {fruit}")

  # 遍历字符串
  word = "Python"
  for char in word:
      print(char)

  # 使用 range()
  for i in range(5):
      print(f"第 {i} 次循环")
  ```

  #### 2.2 `while` 循环

  `while` 循环会一直执行代码块，直到条件为假。

  ```python
  count = 5
  while count > 0:
      print(f"倒计时：{count}")
      count -= 1
  ```

  **无限循环**

  可以通过 `while True` 实现无限循环，但需确保循环内有退出条件。

  ```python
  while True:
      command = input("输入 'exit' 退出：")
      if command == "exit":
          print("程序结束。")
          break
  ```

  ---

  ### 3. 控制循环的特殊语句

  **`break`：退出当前循环**

  ```python
  for i in range(10):
      if i == 5:
          break
      print(i)
  # 输出: 0 1 2 3 4
  ```

  **`continue`：跳过当前循环的剩余部分**

  ```python
  for i in range(10):
      if i % 2 == 0:
          continue
      print(i)
  # 输出: 1 3 5 7 9
  ```

  **`pass`：占位符，不执行任何操作**

  ```python
  for i in range(5):
      if i == 3:
          pass  # 占位符
      print(i)
  ```

  **`else` 和循环搭配使用**

  循环的 `else` 块会在循环正常结束时执行（未被 `break` 打断）。

  ```python
  for i in range(5):
      print(i)
  else:
      print("循环正常结束。")

  # 如果有 break，则 else 不执行
  for i in range(5):
      if i == 3:
          break
      print(i)
  else:
      print("这行不会被打印。")
  ```

  ---

  ### 4. 流程控制的综合实例

  **猜数字游戏**

  ```python
  import random

  # 随机生成一个 1 到 100 的数字
  secret_number = random.randint(1, 100)

  while True:
      guess = int(input("猜一个数字（1-100）："))
      if guess < secret_number:
          print("太小了！")
      elif guess > secret_number:
          print("太大了！")
      else:
          print("恭喜你，猜对了！")
          break
  ```

  **打印乘法表**

  ```python
  # 打印 1 到 9 的乘法表
  for i in range(1, 10):
      for j in range(1, i + 1):
          print(f"{i} * {j} = {i * j}", end="\t")
      print()
  ```

  ---

  ### 5. 提高代码清晰度的建议

  - 避免过深的嵌套，通过提前返回或使用逻辑运算符优化代码。
  - 使用 `else` 和循环结合可以实现特定条件后的操作，但需要谨慎使用。
  - 对于需要退出多层循环的情况，可以使用标志变量或函数封装。

  ---

## 八、函数：模块化的基础

  函数是 Python 中用于将代码逻辑封装为可复用的模块化单元。通过定义函数，可以提高代码的可读性、复用性和维护性。

  ---

  ### 1. 定义和调用函数

  **基本语法**

  使用关键字 `def` 定义函数，函数名应遵循命名规范（小写字母，单词间用下划线分隔）。

  ```python
  # 定义一个简单的函数
  def greet():
      print("Hello, world!")

  # 调用函数
  greet()
  ```

  **带参数的函数**

  参数用于将值传递给函数。

  ```python
  def greet(name):
      print(f"Hello, {name}!")

  greet("Alice")  # 输出: Hello, Alice!
  ```

  **返回值**

  使用 `return` 返回函数的结果。

  ```python
  def add(a, b):
      return a + b

  result = add(5, 3)
  print(result)  # 输出: 8
  ```

  ---

  ### 2. 参数详解

  **默认参数**

  为参数设置默认值，当调用函数时未提供该参数时会使用默认值。

  ```python
  def greet(name="world"):
      print(f"Hello, {name}!")

  greet()          # 输出: Hello, world!
  greet("Alice")   # 输出: Hello, Alice!
  ```

  **位置参数**

  按照定义顺序传递值。

  ```python
  def multiply(a, b):
      return a * b

  print(multiply(3, 4))  # 输出: 12
  ```

  **关键字参数**

  通过参数名传递值，无需遵循顺序。

  ```python
  def describe_pet(animal_type, pet_name):
      print(f"I have a {animal_type} named {pet_name}.")

  describe_pet(pet_name="Buddy", animal_type="dog")
  # 输出: I have a dog named Buddy.
  ```

  **可变参数**

  使用 `*args` 和 `**kwargs` 处理可变数量的参数。

  ```python
  # 接受任意多个位置参数
  def sum_numbers(*args):
      return sum(args)

  print(sum_numbers(1, 2, 3, 4))  # 输出: 10

  # 接受任意多个关键字参数
  def print_details(**kwargs):
      for key, value in kwargs.items():
          print(f"{key}: {value}")

  print_details(name="Alice", age=25)
  # 输出:
  # name: Alice
  # age: 25
  ```

  ---

  ### 3. 作用域与嵌套函数

  **局部变量和全局变量**

  变量的作用范围由其定义的位置决定。

  ```python
  x = 10  # 全局变量

  def func():
      x = 5  # 局部变量
      print(x)  # 输出: 5

  func()
  print(x)  # 输出: 10
  ```

  **使用 `global` 修改全局变量**

  ```python
  x = 10

  def update_global():
      global x
      x = 20

  update_global()
  print(x)  # 输出: 20
  ```

  **嵌套函数与 `nonlocal`**

  嵌套函数可以访问外部函数的变量，使用 `nonlocal` 修改外层函数变量。

  ```python
  def outer():
      x = 10

      def inner():
          nonlocal x
          x = 20
          print(f"Inner x: {x}")

      inner()
      print(f"Outer x: {x}")

  outer()
  # 输出:
  # Inner x: 20
  # Outer x: 20
  ```

  ---

  ### 4. Lambda 表达式

  Lambda 表达式是一种简洁的匿名函数，用于定义简单的函数。

  ```python
  # 普通函数
  def square(x):
      return x ** 2

  # Lambda 表达式
  square_lambda = lambda x: x ** 2

  print(square(4))         # 输出: 16
  print(square_lambda(4))  # 输出: 16
  ```

  Lambda 表达式常与内置函数如 `map`、`filter`、`sorted` 等结合使用。

  ```python
  numbers = [1, 2, 3, 4, 5]

  # 使用 map 应用函数
  squared = map(lambda x: x ** 2, numbers)
  print(list(squared))  # 输出: [1, 4, 9, 16, 25]

  # 使用 filter 过滤数据
  evens = filter(lambda x: x % 2 == 0, numbers)
  print(list(evens))  # 输出: [2, 4]
  ```

  ---

  ### 5. 函数的高级特性

  **函数作为参数**

  函数也可以作为参数传递。

  ```python
  def apply_function(func, value):
      return func(value)

  print(apply_function(lambda x: x ** 2, 5))  # 输出: 25
  ```

  **函数作为返回值**

  ```python
  def make_multiplier(factor):
      def multiplier(x):
          return x * factor
      return multiplier

  double = make_multiplier(2)
  print(double(5))  # 输出: 10
  ```

  **装饰器**

  装饰器是一种高级函数，用于动态地修改其他函数的行为。

  ```python
  def decorator(func):
      def wrapper():
          print("Before the function call")
          func()
          print("After the function call")
      return wrapper

  @decorator
  def say_hello():
      print("Hello!")

  say_hello()
  # 输出:
  # Before the function call
  # Hello!
  # After the function call
  ```

  ---

  ### 6. 函数的应用实例

  **计算阶乘**

  ```python
  def factorial(n):
      if n == 0:
          return 1
      return n * factorial(n - 1)

  print(factorial(5))  # 输出: 120
  ```

  **斐波那契数列**

  ```python
  def fibonacci(n):
      if n <= 1:
          return n
      return fibonacci(n - 1) + fibonacci(n - 2)

  for i in range(10):
      print(fibonacci(i), end=" ")  # 输出: 0 1 1 2 3 5 8 13 21 34
  ```

  ---
##  九、文件操作：与外部数据交互的桥梁

  文件操作是 Python 中用于读写外部数据的重要功能。通过内置的文件操作方法，程序可以方便地与文件系统交互。

  ---

  ### 1. 打开和关闭文件

  **基本语法**

  使用 `open()` 函数打开文件，完成操作后使用 `close()` 关闭文件以释放资源。

  ```python
  # 打开文件
  file = open("example.txt", "w")

  # 写入内容
  file.write("Hello, World!")

  # 关闭文件
  file.close()
  ```

  **`with` 语句自动管理资源**

  推荐使用 `with` 语句代替手动关闭文件，避免忘记 `close()`。

  ```python
  with open("example.txt", "w") as file:
      file.write("Hello, Python!")
  # 文件会在 with 块结束时自动关闭
  ```

  ---

  ### 2. 文件模式

  在 `open()` 函数中，通过模式指定文件的操作方式：

  | 模式 | 描述                          |
  |------|-------------------------------|
  | `"r"` | 以只读模式打开（默认）。文件不存在时抛出异常。 |
  | `"w"` | 以写入模式打开。文件存在会清空内容，不存在则创建。 |
  | `"a"` | 以追加模式打开。文件不存在则创建。 |
  | `"x"` | 以创建模式打开。文件已存在则抛出异常。 |
  | `"b"` | 以二进制模式打开。可与其他模式组合，如 `"rb"`。 |
  | `"t"` | 以文本模式打开（默认）。可与其他模式组合，如 `"rt"`。 |

  ---

  ### 3. 读取文件内容

  **使用 `read()`**

  读取文件的全部内容。

  ```python
  with open("example.txt", "r") as file:
      content = file.read()
      print(content)
  ```

  **使用 `readline()`**

  逐行读取文件，每次读取一行。

  ```python
  with open("example.txt", "r") as file:
      line = file.readline()
      while line:
          print(line.strip())  # 去掉换行符
          line = file.readline()
  ```

  **使用 `readlines()`**

  将文件的所有行作为一个列表返回。

  ```python
  with open("example.txt", "r") as file:
      lines = file.readlines()
      for line in lines:
          print(line.strip())
  ```

  ---

  ### 4. 写入文件内容

  **使用 `write()`**

  写入字符串到文件。

  ```python
  with open("example.txt", "w") as file:
      file.write("This is a new line.\n")
      file.write("Another line.")
  ```

  **使用 `writelines()`**

  写入多个字符串（列表形式）。

  ```python
  lines = ["First line\n", "Second line\n", "Third line\n"]
  with open("example.txt", "w") as file:
      file.writelines(lines)
  ```

  ---

  ### 5. 文件指针操作

  文件指针表示文件的当前操作位置。

  **使用 `seek()`**

  移动文件指针到指定位置。

  ```python
  with open("example.txt", "r") as file:
      file.seek(5)  # 移动到第 5 个字节
      print(file.read())  # 读取剩余内容
  ```

  **使用 `tell()`**

  获取当前文件指针位置。

  ```python
  with open("example.txt", "r") as file:
      print(file.tell())  # 输出: 0
      file.read(5)
      print(file.tell())  # 输出: 5
  ```

  ---

  ### 6. 文件的其他操作

  **检查文件是否存在**

  使用 `os` 或 `pathlib` 模块检查文件是否存在。

  ```python
  import os

  if os.path.exists("example.txt"):
      print("文件存在")
  else:
      print("文件不存在")
  ```

  **删除文件**

  ```python
  os.remove("example.txt")
  ```

  **创建目录**

  ```python
  os.mkdir("new_folder")
  ```

  **删除目录**

  ```python
  os.rmdir("new_folder")
  ```

  ---

  ### 7. 二进制文件操作

  用于处理图片、视频等非文本文件。

  ```python
  # 读取二进制文件
  with open("example.png", "rb") as file:
      data = file.read()
      print(data)

  # 写入二进制文件
  with open("example_copy.png", "wb") as file:
      file.write(data)
  ```

  ---

  ### 8. 综合实例

  **统计文件中的单词数**

  ```python
  with open("example.txt", "r") as file:
      content = file.read()
      words = content.split()
      print(f"单词数量: {len(words)}")
  ```

  **合并多个文件的内容**

  ```python
  files = ["file1.txt", "file2.txt", "file3.txt"]

  with open("merged.txt", "w") as outfile:
      for fname in files:
          with open(fname, "r") as infile:
              outfile.write(infile.read())
  ```

  **备份文件**

  ```python
  import shutil

  shutil.copy("example.txt", "example_backup.txt")
  ```
  ---

## 十一、异常处理

  异常处理是 Python 提供的一种机制，用于捕获程序运行时的错误并进行适当的处理，从而防止程序因错误而崩溃。

  ---

  ### 1. 什么是异常？

  异常是程序运行时出现的错误。常见的异常类型包括：

  | 异常类型       | 描述                                  |
  |----------------|---------------------------------------|
  | `ValueError`   | 当传递无效的参数时发生，例如将字符串转为整数时。 |
  | `TypeError`    | 操作或函数应用于不支持的类型时。       |
  | `KeyError`     | 字典中不存在指定键时。                |
  | `IndexError`   | 列表索引超出范围时。                  |
  | `ZeroDivisionError` | 尝试除以零时。                   |
  | `FileNotFoundError` | 文件不存在时。                    |

  ---

  ### 2. 捕获异常

  通过 `try-except` 块捕获异常，防止程序崩溃。

  ```python
  try:
      x = int(input("请输入一个数字："))
      print(10 / x)
  except ZeroDivisionError:
      print("除数不能为零！")
  except ValueError:
      print("请输入一个有效的数字！")
  ```

  ---

  ### 3. 多个异常处理

  可以在一个 `try` 块中捕获多种异常，并分别处理。

  ```python
  try:
      x = int(input("请输入一个数字："))
      print(10 / x)
  except ZeroDivisionError:
      print("除数不能为零！")
  except (ValueError, TypeError):
      print("输入错误，请检查输入！")
  ```

  ---

  ### 4. 捕获所有异常

  使用通配符 `Exception` 捕获所有异常，但建议具体异常优先。

  ```python
  try:
      x = int(input("请输入一个数字："))
      print(10 / x)
  except Exception as e:
      print(f"发生异常：{e}")
  ```

  ---

  ### 5. `else` 和 `finally`

  - **`else`**: 当没有发生异常时执行。
  - **`finally`**: 无论是否发生异常都会执行，常用于释放资源。

  ```python
  try:
      file = open("example.txt", "r")
      content = file.read()
  except FileNotFoundError:
      print("文件不存在！")
  else:
      print(content)
  finally:
      if 'file' in locals() and not file.closed:
          file.close()
      print("文件操作结束。")
  ```

  ---

  ### 6. 自定义异常

  通过继承 `Exception` 类，可以定义自己的异常类型。

  ```python
  class CustomError(Exception):
      pass

  def check_number(num):
      if num < 0:
          raise CustomError("数字不能为负数！")

  try:
      check_number(-1)
  except CustomError as e:
      print(f"自定义异常捕获：{e}")
  ```

  ---

  ### 7. 使用 `assert` 进行断言

  `assert` 用于检查条件是否满足，如果条件为 `False`，抛出 `AssertionError`。

  ```python
  x = 10
  assert x > 0, "x 必须大于 0"
  assert x < 5, "x 必须小于 5"  # 这行会抛出 AssertionError
  ```

  ---

  ### 8. 异常处理的小建议

  1. **优先捕获具体异常**：尽量避免直接捕获 `Exception`。
  2. **处理必要的异常**：仅捕获需要处理的异常，其他异常应允许程序抛出。
  3. **使用 `finally`**：在涉及资源操作（如文件、数据库连接）时，确保资源被正确释放。
  4. **记录日志**：通过 `logging` 模块记录异常信息，便于后续分析。

  ---

  ### 9. 实例

  - 文件读取的异常处理

  ```python
  def read_file(filename):
      try:
          with open(filename, "r") as file:
              return file.read()
      except FileNotFoundError:
          print(f"文件 {filename} 不存在！")
      except PermissionError:
          print(f"没有权限读取文件 {filename}！")
      except Exception as e:
          print(f"发生未知错误：{e}")

  content = read_file("example.txt")
  ```

  - 网络请求的异常处理

  ```python
  import requests

  def fetch_data(url):
      try:
          response = requests.get(url)
          response.raise_for_status()  # 检查 HTTP 状态码
          return response.json()
      except requests.exceptions.HTTPError as e:
          print(f"HTTP 错误: {e}")
      except requests.exceptions.ConnectionError:
          print("网络连接错误！")
      except Exception as e:
          print(f"发生未知错误: {e}")

  data = fetch_data("https://jsonplaceholder.typicode.com/posts")
  ```

  ---

## 十二、模块与包

  模块和包是 Python 中用于组织代码的重要工具。通过模块和包，开发者可以将代码划分为多个文件和目录，从而提高代码的可维护性和复用性。

  ---

  ### 1. 什么是模块？

  模块是一个 Python 文件，包含函数、类和变量的定义，也可以包括可执行的代码。模块的作用是将代码逻辑分块，便于复用和组织。

  **导入模块**

  使用 `import` 语句导入模块。

  ```python
  # 导入内置模块
  import math

  # 使用模块中的函数
  print(math.sqrt(16))  # 输出: 4.0
  ```

  **自定义模块**

  创建一个 `mymodule.py` 文件，定义一个函数：

  ```python
  # 文件: mymodule.py
  def greet(name):
      print(f"Hello, {name}!")
  ```

  在另一个文件中导入并使用：

  ```python
  import mymodule

  mymodule.greet("Alice")  # 输出: Hello, Alice!
  ```

  **模块的多种导入方式**

  ```python
  # 导入整个模块
  import math

  # 导入模块中的特定函数或变量
  from math import sqrt, pi

  # 导入并重命名模块
  import math as m

  # 使用通配符导入（不推荐）
  from math import *
  ```

  ---

  ### 2. 什么是包？

  包是一个包含多个模块的目录。通过使用包，可以组织和管理大量的模块。包的目录中必须包含一个特殊的 `__init__.py` 文件（Python 3.3 之后可选，但建议保留）。

  **创建包**

  假设目录结构如下：

  ```
  mypackage/
      __init__.py
      module1.py
      module2.py
  ```

  - `__init__.py` 用于标识目录是一个包，可以为空。
  - `module1.py` 和 `module2.py` 是包内的模块。

  **使用包**

  ```python
  # 导入包中的模块
  from mypackage import module1

  module1.some_function()

  # 导入包中的特定函数
  from mypackage.module2 import another_function
  ```

  ---

  ### 3. 搜索模块的路径

  Python 在导入模块时，会按顺序搜索以下路径：

  1. 当前脚本所在目录
  2. `PYTHONPATH` 环境变量指定的目录
  3. Python 的标准库目录
  4. 第三方包的安装目录（如 `site-packages`）

  可以通过 `sys.path` 查看所有搜索路径：

  ```python
  import sys
  print(sys.path)
  ```

  ---

  ### 4. 常用内置模块

  Python 提供了许多内置模块，以下是几个常用模块的示例：

  - **`os`**：文件和目录操作
  - **`sys`**：与 Python 解释器交互
  - **`math`**：数学运算
  - **`random`**：随机数生成
  - **`datetime`**：日期和时间处理

  ```python
  # 使用 os 模块列出当前目录的文件
  import os
  print(os.listdir("."))

  # 使用 random 模块生成随机数
  import random
  print(random.randint(1, 10))
  ```

  ---
## 十三、面向对象编程：组织与封装的艺术

  面向对象编程（OOP）是 Python 中的重要编程范式，通过类和对象来组织代码，使其更具模块化和复用性。

  ---

  ### 1. 什么是面向对象？

  - **类（Class）**：定义对象的蓝图，描述对象的属性和行为。
  - **对象（Object）**：类的实例，是具体的数据和功能的结合体。
  - **主要特性**：
  1. **封装**：将数据和方法打包在对象中。
  2. **继承**：子类继承父类的属性和方法。
  3. **多态**：对象可以根据实际类型表现出不同的行为。

  ---

  ### 2. 定义类和对象

  **创建类**

  ```python
  class Person:
      # 初始化方法
      def __init__(self, name, age):
          self.name = name  # 实例属性
          self.age = age

      # 实例方法
      def greet(self):
          print(f"Hello, my name is {self.name} and I am {self.age} years old.")
  ```

  **创建对象**

  ```python
  # 创建对象
  person1 = Person("Alice", 25)

  # 访问属性
  print(person1.name)  # 输出: Alice

  # 调用方法
  person1.greet()  # 输出: Hello, my name is Alice and I am 25 years old.
  ```

  ---

  ### 3. 类的基本组成

  **属性**

  - **实例属性**：在 `__init__` 方法中定义，属于具体对象。
  - **类属性**：直接在类中定义，属于整个类共享。

  ```python
  class Person:
      species = "Human"  # 类属性

      def __init__(self, name, age):
          self.name = name  # 实例属性
          self.age = age

  # 类属性访问
  print(Person.species)  # 输出: Human

  # 实例属性访问
  person1 = Person("Alice", 25)
  print(person1.name)  # 输出: Alice
  ```

  **方法**

  - **实例方法**：操作实例属性，需传递 `self`。
  - **类方法**：操作类属性，需传递 `cls`，用 `@classmethod` 装饰。
  - **静态方法**：与类或实例无关，用 `@staticmethod` 装饰。

  ```python
  class Example:
      class_attribute = "Class Level"

      def instance_method(self):
          print("This is an instance method.")

      @classmethod
      def class_method(cls):
          print(f"This is a class method. Class attribute: {cls.class_attribute}")

      @staticmethod
      def static_method():
          print("This is a static method.")

  # 调用方法
  example = Example()
  example.instance_method()
  Example.class_method()
  Example.static_method()
  ```

  ---

  ### 4. 继承与多态

  **继承**

  子类可以继承父类的属性和方法。

  ```python
  class Animal:
      def speak(self):
          print("I am an animal.")

  class Dog(Animal):
      def speak(self):
          print("Woof! Woof!")

  # 创建对象
  dog = Dog()
  dog.speak()  # 输出: Woof! Woof!
  ```

  **使用 `super()` 调用父类方法**

  ```python
  class Animal:
      def __init__(self, name):
          self.name = name

  class Dog(Animal):
      def __init__(self, name, breed):
          super().__init__(name)  # 调用父类构造函数
          self.breed = breed

  dog = Dog("Buddy", "Golden Retriever")
  print(dog.name, dog.breed)  # 输出: Buddy Golden Retriever
  ```

  **多态**

  多态允许不同类的对象以统一的方式调用相同的方法。

  ```python
  class Cat(Animal):
      def speak(self):
          print("Meow!")

  animals = [Dog(), Cat()]

  for animal in animals:
      animal.speak()
  # 输出:
  # Woof! Woof!
  # Meow!
  ```

  ---

  ### 5. 封装与私有属性

  通过双下划线 `__` 定义私有属性，仅限类内部访问。

  ```python
  class BankAccount:
      def __init__(self, balance):
          self.__balance = balance  # 私有属性

      def deposit(self, amount):
          self.__balance += amount

      def get_balance(self):
          return self.__balance

  account = BankAccount(1000)
  account.deposit(500)
  print(account.get_balance())  # 输出: 1500
  # print(account.__balance)  # AttributeError: 'BankAccount' object has no attribute '__balance'
  ```

  ---

  ### 6. 属性和方法的高级用法

  **属性装饰器**

  使用 `@property` 将方法变为可访问的属性。

  ```python
  class Circle:
      def __init__(self, radius):
          self.radius = radius

      @property
      def area(self):
          return 3.14 * self.radius ** 2

  circle = Circle(5)
  print(circle.area)  # 输出: 78.5
  ```

  **魔术方法**

  魔术方法用于实现对象的特殊行为，如加法、字符串表示等。

  ```python
  class Vector:
      def __init__(self, x, y):
          self.x = x
          self.y = y

      def __add__(self, other):
          return Vector(self.x + other.x, self.y + other.y)

      def __str__(self):
          return f"Vector({self.x}, {self.y})"

  v1 = Vector(1, 2)
  v2 = Vector(3, 4)
  print(v1 + v2)  # 输出: Vector(4, 6)
  ```

  ---

  ### 7. 综合实例

  **简单的学生管理系统**

  ```python
  class Student:
      def __init__(self, name, age, grade):
          self.name = name
          self.age = age
          self.grade = grade

      def get_details(self):
          return f"Name: {self.name}, Age: {self.age}, Grade: {self.grade}"

  class Classroom:
      def __init__(self):
          self.students = []

      def add_student(self, student):
          self.students.append(student)

      def show_students(self):
          for student in self.students:
              print(student.get_details())

  # 使用示例
  s1 = Student("Alice", 20, "A")
  s2 = Student("Bob", 21, "B")
  classroom = Classroom()
  classroom.add_student(s1)
  classroom.add_student(s2)
  classroom.show_students()
  # 输出:
  # Name: Alice, Age: 20, Grade: A
  # Name: Bob, Age: 21, Grade: B
  ```

  ---

## 本文参考：

- 菜鸟教程：https://www.runoob.com/python3/python3-tutorial.html
- GitHub项目：https://github.com/AccumulateMore/Python 及 https://github.com/jackfrued/Python-100-Days
- B站视频：https://www.bilibili.com/video/BV1wD4y1o7AS/?spm_id_from=333.1387.favlist.content.click&vd_source=0b1d223da4db87ea07b59e8ae8ca4f45
