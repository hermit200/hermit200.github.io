 **文档使用说明**

Anaconda 是一个开源的 Python 和 R 数据科学平台，它通过**Conda 包管理器**提供了强大且便捷的虚拟环境支持，帮助用户在不同项目中隔离 Python 版本和依赖库。用户可以为每个项目创建独立的环境，避免库冲突或兼容性问题。

此文参考 Anaconda 官方文档，会记录一些**常用命令**方便快速配置，和我遇到过的**问题及解决方法**以供查阅

## 1. 安装 
- **下载地址**: [Anaconda 官方下载](https://www.anaconda.com/products/distribution)
- **安装tips**:
 勾选 “Add Anaconda to my PATH environment variable” 以便命令行使用。

---

## 2. 常用命令

  ### 2.1 **环境管理**
  - **创建环境**：
  
  `conda create -n 环境名 python=版本号`
  
  
  - **激活环境**
  
  `conda activate 环境名`
  
  - **退出环境**
  
  `conda deactivate`
  
  - **删除环境**
  
  `conda remove -n 环境名 --all`
  
  - **查看已创建环境**
  
  `conda env list`
  
  ### 2.2 **包管理**
  - **安装包**
  
  下载缓慢时可以用conda forge
  
  ```bash
  conda install 包名
  conda install 包名 = 版本号
  conda install -c conda-forge 包名
  ```
  
  - **更新/卸载/查看已安装包**
  ```bash
  conda update 包名
  conda remove 包名
  conda list
```

## 3.在vscode里使用Anaconda

- **使用命令面板**

  确保 Anaconda 环境已激活，打开 VSCode 的命令面板（**Ctrl + Shift + P**）。
  选择 **Python: Select Interpreter**，在列表中选择对应的 Conda 环境。


- **通过Command Prompt**
  打开 VSCode 终端（**Ctrl + ~**）
  ```bash
  conda activate 环境名
  python your_script.py
  ```

## 4.**我遇见过的问题和解决方法**

- **无法激活环境**

  1. 确认 Conda有没有被配置到系统环境变量中
  
  2. 如果是发生在第一次安装 Conda或者更换终端后，可以尝试`conda init`命令

- **环境冲突**
  报错为：**UnsatisfiableError**，一般是因为不同的包可能有相互冲突的依赖关系
  
  解决方法：
  
  1.创建一个新环境后，指定依赖项的版本
  
  `conda create -n myenv python=3.8 numpy pandas`
  
  也可以使用environment.yml文件来一次性指定多个依赖项
  
  ```yaml
  name: myenv
  dependencies:
    - python=3.8
    - numpy=1.19.2
    - pandas=1.2.3
    - matplotlib
  ```
  
  2.如果使用了forge下载，那么还有可能是因为下载的包来自于不同的Channels而报错
  显示当前频道配置：`conda config --show channels`