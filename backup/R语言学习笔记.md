这篇文章是我的 R 语言学习笔记，涵盖了基础操作、数据处理、可视化技巧以及生信分析等内容，还搭配了一些案例分享，希望能帮到你 😊


---

## **1. R语言基础**

 **1.1 环境与基本操作**
1. **安装与配置**：
   - 下载 R：https://cran.r-project.org/
   - 下载 RStudio：https://www.rstudio.com/products/rstudio/
2. **基础操作**：

 设置工作目录：
     ```R
     getwd() # 查看当前目录
     setwd("你的工作路径") # 设置工作目录
     ```
3.**RStudio工作面板介绍**

![](https://i.ibb.co/gz2bQmq/r-studio.png)

 **1. 脚本编辑区（黄色框）**
**功能**：
- 主要用于编写、编辑和保存 R 语言代码。
- 支持多标签页，可以同时打开和编辑多个脚本文件（例如 `.R`、`.Rmd`）。


**主要按钮**：
- **Run**：运行当前选中的代码行或光标所在行。
- **Source**：运行整个脚本文件。
- **Save**：保存当前脚本文件。
- **Code Tools**：
  - 格式化代码、查找替换等功能。



**2. 环境与历史区（绿色框）**
**功能**：
- **Environment（环境）**：
  - 显示当前 R 会话中加载的所有对象（如数据框、变量、函数）。
  - 提供对象的大小、结构等基本信息。
  - 可以通过点击对象名称快速查看或编辑。

- **History（历史）**：
  - 记录当前 R 会话中运行的所有命令。
  - 可以从历史中选择命令并重新运行。

- **Connections（连接）**：
  - 用于连接数据库或其他外部数据源。

**主要按钮**：
- **Import Dataset**：
  - 提供图形化界面从外部文件（如 CSV、Excel）导入数据。
- **清理环境**：
  - 点击扫帚图标清除所有已加载的变量和对象。





 **3. 绘图与工具区（红色框）**
**功能**：
- **Plots（图形）**：
  - 显示使用 R 代码生成的图表，例如 `ggplot2` 或 `base` 绘图。
  - 提供缩放、导出（为 PDF、PNG 等格式）的功能。
  - 可通过前后按钮浏览之前生成的图表。

- **Files（文件）**：
  - 显示当前工作目录的文件列表，支持文件浏览、打开和删除操作。

- **Packages（包管理器）**：
  - 查看已安装的 R 包。
  - 支持搜索、安装和更新 R 包。

- **Help（帮助文档）**：
  - 显示 R 函数、包的官方文档。
  - 可以通过搜索框快速查找帮助信息。

- **Viewer**：
  - 用于显示 HTML 或交互式的内容（例如 Shiny 应用）。

**主要按钮**：
- **Zoom**：将当前图表放大到新窗口中。
- **Export**：将当前图表导出为图片或 PDF。



 **4. 控制台与终端区（紫色框）**
**功能**：
- **Console（控制台）**：
  - 提供交互式 R 命令行，可以直接运行代码并查看输出结果。
  - 实时显示代码的输出、错误和警告信息。
  - 支持快捷键运行代码，快速测试和调试。

- **Terminal（终端）**：
  - 集成系统终端（如 Bash），可执行非 R 语言相关的系统命令。
  - 适合管理文件、运行 Git 命令等。

- **Background Jobs（后台任务）**：
  - 显示和管理正在运行的后台任务。

**主要按钮**：
- **清空控制台**：清除控制台中现有的输出内容。
- **运行按钮**：运行选中的代码或脚本。




**总结**
| **标记** | **窗口名称**          | **主要功能**                                                                 |
|----------|-----------------------|------------------------------------------------------------------------------|
| **1**    | 脚本编辑区            | 编写、编辑 R 代码，运行和保存脚本文件。                                       |
| **2**    | 环境与历史区          | 管理数据对象，查看运行历史，导入外部数据。                                     |
| **3**    | 绘图与工具区          | 显示图表，管理文件与包，查看帮助文档。                                         |
| **4**    | 控制台与终端区        | 运行 R 代码，显示输出和错误信息，执行系统命令。                                 |




 **1.2 数据类型与结构**

**1.2.1 基本数据类型**
- **数值型**（Numeric）：
  ```R
  x <- 10.5
  class(x) # 输出："numeric"
  ```
- **字符型**（Character）：
  ```R
  name <- "R语言"
  class(name) # 输出："character"
  ```
- **逻辑型**（Logical）：
  ```R
  is_true <- TRUE
  class(is_true) # 输出："logical"
  ```
- **因子**（Factor）：用于表示分类变量。
  ```R
  colors <- factor(c("红", "蓝", "蓝", "红"))
  class(colors) # 输出："factor"
  ```

**1.2.2 数据结构**
1. **向量**（Vector）：同类型元素的集合。
   ```R
   vec <- c(1, 2, 3, 4)
   ```
2. **矩阵**（Matrix）：二维的数值集合。
   ```R
   mat <- matrix(1:6, nrow = 2, ncol = 3)
   ```
3. **数据框**（Data Frame）：表格形式的结构，类似于 Excel。
   ```R
   df <- data.frame(ID = 1:3, Name = c("A", "B", "C"))
   ```
4. **列表**（List）：可以包含不同类型的元素。
   ```R
   lst <- list(Name = "R", Numbers = c(1, 2, 3))
   ```



**1.3 基础操作**

1. **变量赋值**：
   ```R
   x <- 5
   y <- c(1, 2, 3)
   ```
2. **基本函数**：
   - 统计函数：
     ```R
     mean(y) # 平均值
     sum(y)  # 总和
     length(y) # 长度
     ```
   - 序列生成：
     ```R
     seq(1, 10, by = 2) # 输出：1, 3, 5, 7, 9
     ```
3. **条件语句**：
   ```R
   if (x > 3) {
     print("x大于3")
   } else {
     print("x小于等于3")
   }
   ```
4. **循环语句**：
   ```R
   for (i in 1:5) {
     print(i)
   }
   ```



## **2. 数据操作**

**2.1 数据导入与导出**

1. **读取 CSV 文件**：
   ```R
   data <- read.csv("data.csv")
   head(data) # 查看前几行
   ```
2. **保存数据**：
   ```R
   write.csv(data, "output.csv")
   ```
3. **读取 Excel 文件**：

   - 安装 `readxl` 包：
  
     ```R
     install.packages("readxl")
     library(readxl)
     data <- read_excel("data.xlsx")
     ```



**2.2 数据清洗**

1. **筛选与过滤**：
   ```R
   subset(data, Age > 30) # 筛选年龄大于30的行
   ```
2. **缺失值处理**：
   - 检查缺失值：
     ```R
     is.na(data)
     ```
   - 移除缺失值：
     ```R
     na.omit(data)
     ```
3. **排序与重排**：
   ```R
   sorted_data <- data[order(data$Age), ] # 按年龄排序
   ```



**2.3 数据操作包**

**2.3.1 dplyr 包核心函数**
```R
install.packages("dplyr")
library(dplyr)

# 示例数据
data <- data.frame(ID = 1:5, Age = c(25, 30, 35, 40, 45), Income = c(50, 60, 70, 80, 90))

# 筛选行
data %>% filter(Age > 30)

# 选择列
data %>% select(ID, Age)

# 增加新列
data %>% mutate(Savings = Income - 20)

# 分组与聚合
data %>% group_by(Age) %>% summarize(Mean_Income = mean(Income))
```



## **3. 数据可视化**

**3.1 基础绘图**

1. **散点图**：
   ```R
   x <- 1:10
   y <- x^2
   plot(x, y, type = "p", main = "散点图", xlab = "X值", ylab = "Y值")
   ```

2. **直方图**：
   ```R
   hist(mtcars$mpg, main = "直方图", xlab = "mpg", col = "lightblue")
   ```



**3.2 ggplot2 高级可视化**


**基本概念**

ggplot2 的核心是图层（layers）概念，每个图表由以下几部分组成：

数据：用 data 指定数据框。
美学映射（Aesthetic Mapping）：用 aes() 定义变量与图形属性（如颜色、大小）的映射。
几何对象（Geometries）：用 geom_*() 指定图形类型（如点、线、柱状图等）。
图层叠加：可以叠加多个 geom_* 图层，生成复杂图形。

**基本绘图模板**

```R
ggplot(data = 数据框, aes(x = 变量1, y = 变量2)) +
  geom_几何对象() +
  其他图层
```

**示例**
```R
# 加载必要的包
library(ggplot2)

# 模拟城市数据集
set.seed(42)
data <- data.frame(
  City = paste("City", 1:20),                # 城市名称
  Population = runif(20, 500000, 10000000), # 城市人口 (50 万 - 1000 万)
  GDP = runif(20, 10000, 90000),            # 城市 GDP (1 万 - 9 万美元)
  UnemploymentRate = runif(20, 2, 15)       # 失业率 (2% - 15%)
)

# 绘制散点图
ggplot(data, aes(x = Population / 1e6,      # 人口 (单位：百万)
                 y = GDP,                   # GDP
                 size = UnemploymentRate,  # 点大小为失业率
                 color = UnemploymentRate)) +  # 点颜色为失业率
  geom_point(alpha = 0.8, shape = 21) +     # 半透明点，带边框
  scale_color_gradient(low = "green", high = "red", name = "Unemployment Rate (%)") + # 渐变颜色
  scale_size(range = c(3, 12), name = "Unemployment Rate (%)") + # 点大小范围
  labs(title = "City Population vs GDP",
       subtitle = "A visual exploration of urban population, GDP, and unemployment rate",
       x = "Population (Million)",
       y = "GDP (USD)") + 
  theme_minimal(base_size = 15) +           # 简洁主题
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18), # 居中标题
    plot.subtitle = element_text(hjust = 0.5, size = 14),            # 居中副标题
    axis.text = element_text(size = 12),                             # 坐标轴标签大小
    axis.title = element_text(size = 14, face = "bold"),             # 坐标轴标题样式
    legend.position = "right",                                       # 图例放置右侧
    panel.grid.minor = element_blank()                               # 移除次网格线
  )

```

**图片展示**

![](https://i.ibb.co/QDykb3P/example-Forggplot.png)

## **4. 统计分析**

**4.1 描述性统计**
```R
data <- c(10, 20, 30, 40, 50)
mean(data) # 平均值
median(data) # 中位数
sd(data) # 标准差
summary(data) # 数据分布
```



**4.2 假设检验**
1. **单样本 t 检验**：
   ```R
   t.test(data, mu = 25)
   ```
2. **双样本 t 检验**：
   ```R
   t.test(data1, data2, paired = TRUE)
   ```

## **5.Bioconductor,用于生物信息学分析的开源项目**

**5.1什么是 Bioconductor？**

**Bioconductor** 是一个用于生物信息学分析的开源软件项目，构建在 R 语言之上，专注于基因组学、生物医学研究等领域的数据分析。它提供了大量的 R 包，支持从数据处理到高层次分析的全流程，例如基因表达分析、功能富集分析、基因组可视化等。

官方网站：[Bioconductor](https://www.bioconductor.org/)



**5.2 Bioconductor 的功能**
1. **丰富的 R 包**：
   - Bioconductor 提供超过 2000 个 R 包，涵盖从基础数据处理到高级分析的各个方面。
   - 常见领域包括转录组学、蛋白组学、功能注释、代谢组学等。

2. **支持多种生物数据格式**：
   - 支持处理常见的生物数据文件格式（如 FASTQ、BAM、VCF、GTF 等）。
   - 提供与公共生物数据库（如 GEO、Ensembl、KEGG）的交互接口。

3. **标准化分析流程**：
   - 提供多个包，用于标准化数据预处理、统计分析、可视化和功能富集分析。

4. **社区支持**：
   - 拥有活跃的开发者社区，提供最新的生物信息学工具和分析方法。



**5.3功能富集分析**
功能富集分析（Functional Enrichment Analysis）是 Bioconductor 的一个常见应用场景，用于分析基因列表是否显著富集于某些生物学功能、通路或注释中。

**主要分析目标**
- **GO（Gene Ontology）分析**：评估基因列表是否富集于某些 GO term（如生物过程、分子功能或细胞组分）。
- **KEGG 通路分析**：检查基因列表是否显著关联于某些 KEGG 通路。
- **Reactome 分析**：探索基因列表在 Reactome 数据库中的生物通路。




**功能富集分析的常用 Bioconductor 包**

| **包名称**          | **功能**                                                                                          |
|---------------------|--------------------------------------------------------------------------------------------------|
| **clusterProfiler** | 支持 GO、KEGG、Reactome 等功能富集分析与可视化，是人类功能富集分析的核心工具之一。                  |
| **org.Hs.eg.db**    | 人类基因注释数据库，用于将基因 ID 映射到 GO、KEGG 等注释。                                         |
| **KEGGREST**        | 提供 KEGG 数据库的直接接口，用于 KEGG 通路的查询与分析。                                            |



**功能富集分析的基本步骤**

**1. 安装必要的包**
```R
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# 安装 Bioconductor 包
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "KEGGREST", "pathview"))
```



**2. 准备基因列表**

基因列表通常来自差异表达分析的结果，表示一组感兴趣的基因（例如上调或下调基因）。人类基因的常见 ID 格式包括 **Entrez ID**、**Gene Symbol** 等。

**示例：**
```R
# 示例基因列表 (Entrez ID)
gene_list <- c("1956", "2064", "5290", "5921", "7157")  # 这些 ID 为人类基因
```

如果目标基因 ID 不是 Entrez ID，可以使用 `org.Hs.eg.db` 将 Gene Symbol 或其他 ID 转换为 Entrez ID：
```R
library(org.Hs.eg.db)
library(clusterProfiler)

# 基因符号 (Gene Symbol) 转换为 Entrez ID
gene_symbols <- c("TP53", "EGFR", "BRCA1", "BRCA2", "MYC")
entrez_ids <- bitr(gene_symbols, 
                   fromType = "SYMBOL", 
                   toType = "ENTREZID", 
                   OrgDb = org.Hs.eg.db)
print(entrez_ids)
```



**3. GO 富集分析**
使用 `clusterProfiler` 包进行 GO 富集分析（包括 Biological Process、Molecular Function、Cellular Component）。

```R
# 加载 clusterProfiler
library(clusterProfiler)

# 使用 enrichGO 进行 GO 富集分析
go_enrich <- enrichGO(gene         = gene_list,
                      OrgDb        = org.Hs.eg.db,  # 使用人类基因注释数据库
                      keyType      = "ENTREZID",    # 输入基因 ID 的类型
                      ont          = "BP",          # 生物过程 (Biological Process)
                      pAdjustMethod = "BH",         # 调整 p 值的方法
                      pvalueCutoff = 0.05,
                      qvalueCutoff = 0.2)

# 查看结果
head(go_enrich)

# 可视化 GO 富集结果
library(enrichplot)
barplot(go_enrich, showCategory = 10)  # 柱状图
dotplot(go_enrich)                     # 点图
```
**结果**

![](https://i.ibb.co/Tm4gmCm/GO.png)


**4. KEGG 富集分析**
使用 `clusterProfiler` 进行 KEGG 通路富集分析，`hsa` 是人类的 KEGG Organism Code。

```R
# 运行 KEGG 富集分析
kegg_enrich <- enrichKEGG(gene         = gene_list,
                          organism     = "hsa",    # 人类的 KEGG 代码
                          keyType      = "kegg",   # 输入基因 ID 类型为 KEGG ID
                          pAdjustMethod = "BH",    # 使用 Benjamini-Hochberg 方法调整 p 值
                          pvalueCutoff = 0.05,
                          qvalueCutoff = 0.2)

# 查看 KEGG 富集分析结果
head(kegg_enrich)

# 可视化 KEGG 富集结果
dotplot(kegg_enrich)    # 点图
```

**结果**
![](https://i.ibb.co/Cw4QN5k/KNEE.png)



