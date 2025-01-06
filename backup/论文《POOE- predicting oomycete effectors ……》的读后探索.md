

本文参考了论文《POOE: predicting oomycete effectors based on a pre-trained large protein language model.》的模型构建方法和数据，并进行了一定的创新和尝试

**写在前面**
因为目前我还只有一台自用的4080显卡，所以此次数据量很小，正负数据集样本各549条，在某些比较难跑的模型里还进行了一定的处理（例如设置max_length小于1000）。

因此，训练出的模型非常容易过拟合、或出现类别不平衡现象。例如当正负样本数量一致时，模型可能会陷入“随机猜测”的状态，或倾向于预测为某一类别。

增大数据集应该会解决该问题，我有机会的话会回来再尝试

**原文献模型简单介绍**


文章流程图请见：https://journals.asm.org/cms/10.1128/msystems.01004-23/asset/6b29f1b9-b98d-4063-b2f4-aeb4e412572e/assets/images/large/msystems.01004-23.f001.jpg

如图，文章大致分为4步。
 一、处理数据
从 **NCBI** 和 **Ensembl** 数据库中下载与阳性样本对应的八种物种的蛋白质组数据，同时通过数据过滤生成阴性样本。将数据集划分为 **80% 用于五重交叉验证的训练集**，**20% 用于独立测试集**。

 二、提取特征
采用 **预训练蛋白质语言模型 ProtTrans** 提取蛋白质序列的深度学习特征，用于下游任务。

三、训练模型
使用 **支持向量机（SVM）** 进行 **五重交叉验证和独立测试**，对模型性能进行评估。

四、纵向比较模型表现
- **序列编码方案**：CT、DPC、PSSM、Doc2Vec 和 ESM。
- **机器学习算法**：随机森林（RF）、AdaBoost，以及深度学习算法 CNN。
- **现有的卵菌效应子预测方法**：EffectorO、EffectorP3.0 和 deepredeff。
- **基于序列/基序的效应子识别策略**：BLAST 和 FIMO。

**下面是基于此篇文章的部分数据，作者进行的探索内容的展示：**

## 1.使用蛋白预训练模型提取数据

### 1.ESM系列 (Evolutionary Scale Modeling，https://github.com/facebookresearch/esm）

**ESM系列模型基于深度学习中的Transformer架构**，主要利用自注意力机制来捕获蛋白质序列中的局部和全局关系。模型通过大规模蛋白质序列数据进行无监督预训练，从而学习到能够有效预测蛋白质结构和功能的高质量序列表征。



以下代码主要采用**esm2_t33_650M_UR50D**模型，然后用随机森林进行5折交叉验证

```python

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.utils import shuffle
from Bio import SeqIO
import torch
import esm
import matplotlib.pyplot as plt
import seaborn as sns


# 加载 ESM2 模型
def load_esm2_model(model_name="esm2_t33_650M_UR50D"):
"""
加载 ESM2 模型并初始化设备。
:param model_name: ESM2 模型名称
:return: 模型, batch_converter, 设备
"""
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
batch_converter = alphabet.get_batch_converter()
model.eval()  # 设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
return model, batch_converter, device


# 处理FASTA并裁切
def parse_fasta(in_file, max_length=500):
"""
从FASTA文件中读取序列，并裁剪为最大长度。
:param in_file: 输入FASTA文件路径
:param max_length: 最大序列长度
:return: 裁剪后的序列字典
"""
seqs = dict()
for i in SeqIO.parse(in_file, "fasta"):
if "|" in i.id:
k = i.id.split("|")[1]
if ":" in k:
k = k.split(":")[0]
else:
k = i.id
v = str(i.seq)
# 裁切序列到 max_length
if len(v) > max_length:
v = v[:max_length]
seqs[k] = v
return seqs


# 提取蛋白序列特征 (ESM2)
def extract_esm2_features(sequences, model, batch_converter, device, batch_size=16):
"""
使用ESM2模型提取蛋白序列的特征。
:param sequences: 序列字典
:param model: 预训练模型
:param batch_converter: 批量转换器
:param device: 设备
:param batch_size: 批量大小
:return: 提取的特征矩阵
"""
seq_list = [(k, v) for k, v in sequences.items()]  # 转换为列表
embeddings = []

for i in range(0, len(seq_list), batch_size):
batch_data = seq_list[i : i + batch_size]  # 按批量获取序列
batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)  # 格式化数据
batch_tokens = batch_tokens.to(device)

with torch.no_grad():
results = model(batch_tokens, repr_layers=[33], return_contacts=False)

batch_embeddings = results["representations"][33].mean(1).cpu().numpy()  # 平均池化
embeddings.append(batch_embeddings)

return np.vstack(embeddings)


# 数据预处理
def prepare_data(pos_fasta, neg_fasta, model, batch_converter, device, batch_size=16, max_length=500):
"""
预处理正负样本的FASTA文件，并提取特征。
:param pos_fasta: 正样本FASTA文件路径
:param neg_fasta: 负样本FASTA文件路径
:param model: 预训练模型
:param batch_converter: 批量转换器
:param device: 设备
:param batch_size: 批量大小
:param max_length: 最大序列长度
:return: 特征和标签
"""
pos_sequences = parse_fasta(pos_fasta, max_length=max_length)
neg_sequences = parse_fasta(neg_fasta, max_length=max_length)

pos_features = extract_esm2_features(pos_sequences, model, batch_converter, device, batch_size=batch_size)
neg_features = extract_esm2_features(neg_sequences, model, batch_converter, device, batch_size=batch_size)

pos_labels = np.ones(len(pos_features))
neg_labels = np.zeros(len(neg_features))
features = np.vstack((pos_features, neg_features))
labels = np.concatenate((pos_labels, neg_labels))

features, labels = shuffle(features, labels, random_state=42)
scaler = StandardScaler()
features = scaler.fit_transform(features)
return features, labels


# 绘制混淆矩阵热图
def plot_confusion_matrix(cm, labels):
"""
绘制混淆矩阵的热图。
:param cm: 混淆矩阵
:param labels: 标签类别
"""
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 绘制ROC曲线
def plot_roc_curve(y_test, y_prob):
"""
绘制ROC曲线。
:param y_test: 实际标签
:param y_prob: 预测概率
"""
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()


# 随机森林 + 五折交叉验证
def random_forest_cross_validation(features, labels):
"""
使用随机森林进行五折交叉验证，并评估性能。
:param features: 特征矩阵
:param labels: 标签
"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(
n_estimators=500, max_depth=30, class_weight="balanced", random_state=42
)

scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "confusion_matrix": []}

for train_idx, test_idx in skf.split(features, labels):
X_train, X_test = features[train_idx], features[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

scores["accuracy"].append(accuracy_score(y_test, y_pred))
scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
scores["auc"].append(roc_auc_score(y_test, y_prob))
scores["confusion_matrix"].append(confusion_matrix(y_test, y_pred))

# 绘制ROC曲线和混淆矩阵
plot_roc_curve(y_test, y_prob)
plot_confusion_matrix(confusion_matrix(y_test, y_pred), labels=["Negative", "Positive"])

for metric, values in scores.items():
if metric == "confusion_matrix":
print("Confusion Matrices for each fold:")
for i, cm in enumerate(values):
print(f"Fold {i + 1}:\n{cm}")
else:
print(f"{metric.capitalize()} scores: {values}")
print(f"Average {metric.capitalize()}: {np.mean(values):.4f}")


def main():
base_path = "E:/vscode/code"
pos_fasta = os.path.join(base_path, "positivedata549.fasta")
neg_fasta = os.path.join(base_path, "negativedata549.fasta")

print("Loading ESM2 model...")
model_name = "esm2_t33_650M_UR50D"
model, batch_converter, device = load_esm2_model(model_name)

if device.type == "cuda":
print("Extracting features...")
features, labels = prepare_data(pos_fasta, neg_fasta, model, batch_converter, device, batch_size=16, max_length=500)

print("Training Random Forest with 5-fold cross-validation...")
random_forest_cross_validation(features, labels)
else:
print("CUDA is not available. ")


if __name__ == "__main__":
main()

```

以下是其中一次的结果图片：

![](https://i.ibb.co/gtdL18F/esm2.png)

![](https://i.ibb.co/wL67WTw/esm2-2.png)


### 2.ProtTrans（https://github.com/agemagician/ProtTrans）

**ProtTrans是一系列基于Transformer技术的模型的集合，包括ProtBert和ProtT5等**。这些模型通过Transformer架构进行预训练。
ProtBert是基于BERT模型的蛋白质序列的预训练模型。类似于在自然语言处理中使用的BERT模型，ProtBert利用自监督学习方法来捕捉蛋白质序列中的复杂模式。
ProT5是基于T5 (Text-To-Text Transfer Transformer) 模型进行蛋白质序列预训练的模型。T5模型是一种通用的自编码器-解码器结构，ProT5将这种方法应用于蛋白质序列分析，目的是捕捉序列之间的长距离依赖关系和复杂的功能模式 


以下代码主要采用**Rostlab/prot_bert**模型，然后用随机森林进行5折交叉验证

```python
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.utils import shuffle
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# 加载 ProtTrans 模型
def load_prottrans_model(model_name="Rostlab/prot_bert"):
"""
加载ProtTrans模型和分词器。
:param model_name: ProtTrans模型名称
:return: 模型和分词器
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
return model, tokenizer


# 解析 FASTA 文件
def parse_fasta(in_file):
"""
解析FASTA文件，提取序列。
:param in_file: FASTA文件路径
:return: 序列字典
"""
seqs = dict()
for i in SeqIO.parse(in_file, "fasta"):
k = i.id.split("|")[1] if "|" in i.id else i.id
seqs[k] = str(i.seq)
return seqs


# 提取 ProtTrans 特征
def extract_prottrans_features(sequences, model, tokenizer, batch_size=16, max_length=1024):
"""
使用 ProtTrans 提取序列特征。
:param sequences: 序列字典
:param model: ProtTrans模型
:param tokenizer: ProtTrans分词器
:param batch_size: 批量大小
:param max_length: 最大序列长度
:return: 特征矩阵
"""
seq_list = list(sequences.values())
embeddings = []

for i in range(0, len(seq_list), batch_size):
batch_seqs = seq_list[i:i + batch_size]
inputs = tokenizer(batch_seqs, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

with torch.no_grad():
outputs = model(**inputs)
batch_embeddings = outputs.last_hidden_state.mean(1).squeeze().numpy()
embeddings.append(batch_embeddings)

return np.vstack(embeddings)


# 数据预处理
def prepare_data(pos_fasta, neg_fasta, model, tokenizer, batch_size=16, max_length=1024):
"""
预处理正负样本数据。
:param pos_fasta: 正样本FASTA文件路径
:param neg_fasta: 负样本FASTA文件路径
:param model: ProtTrans模型
:param tokenizer: ProtTrans分词器
:param batch_size: 批量大小
:param max_length: 最大序列长度
:return: 特征和标签
"""
pos_sequences = parse_fasta(pos_fasta)
neg_sequences = parse_fasta(neg_fasta)

pos_features = extract_prottrans_features(pos_sequences, model, tokenizer, batch_size=batch_size, max_length=max_length)
neg_features = extract_prottrans_features(neg_sequences, model, tokenizer, batch_size=batch_size, max_length=max_length)

pos_labels = np.ones(len(pos_features))
neg_labels = np.zeros(len(neg_features))

features = np.vstack((pos_features, neg_features))
labels = np.concatenate((pos_labels, neg_labels))

features, labels = shuffle(features, labels, random_state=42)
scaler = StandardScaler()
features = scaler.fit_transform(features)

return features, labels


# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
"""
绘制混淆矩阵热图。
:param cm: 混淆矩阵
:param labels: 标签列表
"""
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 可视化 ROC 曲线
def plot_roc_curve(y_true, y_scores):
"""
绘制ROC曲线。
:param y_true: 实际标签
:param y_scores: 预测概率
"""
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()


# 随机森林 + 五折交叉验证
def random_forest_cross_validation(features, labels):
"""
使用随机森林进行五折交叉验证。
:param features: 特征矩阵
:param labels: 标签
"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(
n_estimators=500, max_depth=30, class_weight="balanced", random_state=42
)

scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "confusion_matrix": []}

for train_idx, test_idx in skf.split(features, labels):
X_train, X_test = features[train_idx], features[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

scores["accuracy"].append(accuracy_score(y_test, y_pred))
scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
scores["auc"].append(roc_auc_score(y_test, y_prob))
scores["confusion_matrix"].append(confusion_matrix(y_test, y_pred))

for metric, values in scores.items():
if metric == "confusion_matrix":
print("Confusion Matrices:")
for i, cm in enumerate(values):
print(f"Fold {i + 1}:\n{cm}")
plot_confusion_matrix(cm, ["Negative", "Positive"])
else:
avg_value = np.mean(values)
print(f"{metric.capitalize()} - Scores: {values}, Average: {avg_value:.4f}")
if metric == "auc":
plot_roc_curve(labels, clf.predict_proba(features)[:, 1])


def main():
base_path = "E:/vscode/code"
pos_fasta = os.path.join(base_path, "positivedata549.fasta")
neg_fasta = os.path.join(base_path, "negativedata549.fasta")

print("Loading ProtTrans model...")
model_name = "Rostlab/prot_bert"
model, tokenizer = load_prottrans_model(model_name)

print("Extracting features...")
features, labels = prepare_data(pos_fasta, neg_fasta, model, tokenizer)

print("Training Random Forest with 5-fold cross-validation...")
random_forest_cross_validation(features, labels)


if __name__ == "__main__":
main()


```

**这个模型训练的结果非常差**（见下图），但是论文原文用的就是ProtTrans的模型，表现效果很好。所以推断是这里表现不佳是**数据量太小**导致的

![](https://i.ibb.co/d5vcRcZ/trans1.png)

## 2.蛋白序列提取算法

下面是一些论文中提到的和常见的蛋白序列提取的代码总结：

### 1.CT

CT编码首先将20个氨基酸分为7组，这种分组基于氨基酸的电荷、极性、疏水性等特性。然后使用CT（Conjoint Triad）方法，一个蛋白质序列被表示为连续的三个氨基酸的组合，由于有7个氨基酸组，组合可能性为7 × 7 × 7 = 343种可能的组合。由此，每一个蛋白质可以转换为一个343维的向量，每个维度对应一种三联体组合的出现频率

代码：

```python

def ct(k,seq,label = 0):
aminos = "AGVCDEFILPHNQWKRMSTY"
ct_category = {
'A':'0','G':'0','V':'0',
'C':'1',
'D':'2','E':'2',
'F':'3','I':'3','L':'3','P':'3',
'H':'4','N':'4','Q':'4','W':'4',
'K':'5','R':'5',
'M':'6','S':'6','T':'6','Y':'6' }
ct_count = {i+j+k:0  for i in '0123456' for j in  '0123456' for k in  '0123456'}
seq = "".join([ct_category[i] for i in seq.upper() if i in aminos])

for i in range(len(seq)-2):
ct_count[seq[i:i+3]] +=1
values = np.array(list(ct_count.values())) / (len(seq)-2)
return k,values,label

```

### 2.DPC

类似CT，DPC基于两个连续氨基酸的组合，通过统计整个蛋白质序列中所有可能的二肽组合的频率来提取蛋白特征

```python
def dpc(k,seq,label = 1):
aminos = "AGVCDEFILPHNQWKRMSTY"
dpc_dict = {i+j:0 for i in aminos for j in aminos}
seq = "".join([i for i in seq.upper() if i in aminos])

for i in range(len(seq)-1):
dpc_dict[seq[i:i+2]] += 1
values = np.array(list(dpc_dict.values())) / (len(seq)-1)
return k, values,label
```

### 3.CSKAAP

CSKAAP 编码通过捕捉蛋白质序列中氨基酸的组成特征（Composition）、空间特征（Spatial）和特定氨基酸对特征（K-Amino Acid Pair）来进行特征提取。具体而言，它首先计算蛋白质序列中20种氨基酸的频率（组成特征），然后分析序列中氨基酸间的相对距离模式（空间特征），最后提取固定间隔的氨基酸对出现频率作为氨基酸对特征。由此，每个蛋白质序列可以被转化为一个高维向量

```python

def CSKAAP(name, seq, label=0):
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
ks = [0, 1, 2]  # 间隔 k 
feature_vector = []  

for k in ks:

feature_dict = {aa1 + aa2: 0 for aa1 in amino_acids for aa2 in amino_acids}

# 计算 k-spaced aa对
for i in range(len(seq) - k - 1):
aa_pair = seq[i] + seq[i + k + 1]
if aa_pair in feature_dict:
feature_dict[aa_pair] += 1

# 归一化
total_pairs = sum(feature_dict.values())
if total_pairs > 0:
feature_dict = {key: value / total_pairs for key, value in feature_dict.items()}


feature_vector.extend(list(feature_dict.values()))

return name, feature_vector, label

```

## 其他

### 1.自训练LSTM模型提取蛋白特征

这里本来是想用unirep（https://github.com/churchlab/UniRep）

这也是一个预训练模型，但是该模型太老旧了，采取tensorflow1框架，遇到报错后作者实在不知道从何改起。

考虑到该模型是lstm模型，所以作者自训练了一个lstm提取特征，并用随机森林进行5折交叉验证，代码如下：

```python

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns


# 解析 FASTA 文件
def parse_fasta(fasta_file):
"""
解析FASTA文件，将序列和对应标签提取出来。
:param fasta_file: 输入的FASTA文件路径
:return: 序列列表和标签列表
"""
sequences = []
labels = []
for record in SeqIO.parse(fasta_file, "fasta"):
sequence = str(record.seq)
label = 1 if "positive" in fasta_file else 0  # 根据文件名区分正负样本
sequences.append(sequence)
labels.append(label)
return sequences, labels


# 编码蛋白序列
def encode_sequences(sequences, max_length=1000):
"""
将蛋白质序列编码为固定长度的数字表示。
:param sequences: 蛋白质序列列表
:param max_length: 最大序列长度
:return: 编码后的序列矩阵
"""
amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20种氨基酸
aa_to_index = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}  # 生成映射（1-20）
aa_to_index["X"] = 0  # 未知字符映射为 0

encoded_sequences = []
for seq in sequences:
encoded = [aa_to_index.get(aa, 0) for aa in seq[:max_length]]  # 截断到 max_length
if len(encoded) < max_length:
encoded.extend([0] * (max_length - len(encoded)))  # 补零到固定长度
encoded_sequences.append(encoded)

return np.array(encoded_sequences, dtype=np.int32)


# 定义数据集类
class ProteinDataset(Dataset):
def __init__(self, sequences, labels):
self.sequences = sequences
self.labels = labels

def __len__(self):
return len(self.sequences)

def __getitem__(self, idx):
# 将输入序列从 [max_length] 转为 [max_length, 1]（时间步，特征数）
sequence = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
label = torch.tensor(self.labels[idx], dtype=torch.int64)
return sequence, label


# 定义 LSTM 模型
class LSTMModel(nn.Module):
def __init__(self, input_size, hidden_size, output_size, num_layers=1):
super(LSTMModel, self).__init__()
self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
self.fc = nn.Linear(hidden_size, output_size)

def forward(self, x):
_, (hn, _) = self.lstm(x)
out = self.fc(hn[-1])  # 取最后一层作为输出
return out


# 训练 LSTM 模型
def train_lstm_model(model, dataloader, criterion, optimizer, device, epochs=10):
"""
训练LSTM模型。
:param model: LSTM模型
:param dataloader: 数据加载器
:param criterion: 损失函数
:param optimizer: 优化器
:param device: 设备（CPU或GPU）
:param epochs: 训练轮数
"""
model.train()
for epoch in range(epochs):
epoch_loss = 0
for batch_sequences, batch_labels in dataloader:
batch_sequences = batch_sequences.to(device)
batch_labels = batch_labels.to(device)

optimizer.zero_grad()
outputs = model(batch_sequences)
loss = criterion(outputs, batch_labels.unsqueeze(1).float())
loss.backward()
optimizer.step()
epoch_loss += loss.item()

print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


# 使用 LSTM 提取特征
def extract_lstm_features(model, dataloader, device):
"""
使用训练好的LSTM模型提取特征。
:param model: LSTM模型
:param dataloader: 数据加载器
:param device: 设备（CPU或GPU）
:return: 特征矩阵和标签
"""
model.eval()
features = []
labels = []

with torch.no_grad():
for batch_sequences, batch_labels in dataloader:
batch_sequences = batch_sequences.to(device)
outputs = model(batch_sequences)
features.append(outputs.cpu().numpy())
labels.append(batch_labels.numpy())

return np.vstack(features), np.hstack(labels)


# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 可视化 ROC 曲线
def plot_roc_curve(y_true, y_scores):
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.show()


# 随机森林分类
def random_forest_classification(features, labels):
"""
使用随机森林进行分类。
:param features: 特征矩阵
:param labels: 标签
"""
scaler = StandardScaler()
features = scaler.fit_transform(features)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(features, labels)

predictions = clf.predict(features)
probas = clf.predict_proba(features)[:, 1]

accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, zero_division=0)
recall = recall_score(labels, predictions, zero_division=0)
f1 = f1_score(labels, predictions, zero_division=0)
cm = confusion_matrix(labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 可视化结果
plot_confusion_matrix(cm, ["Negative", "Positive"])
plot_roc_curve(labels, probas)


def main():
base_path = "E:/vscode/code"
pos_fasta = os.path.join(base_path, "positivedata549.fasta")
neg_fasta = os.path.join(base_path, "negativedata549.fasta")

# 解析FASTA文件并编码
pos_sequences, pos_labels = parse_fasta(pos_fasta)
neg_sequences, neg_labels = parse_fasta(neg_fasta)
sequences = pos_sequences + neg_sequences
labels = pos_labels + neg_labels
encoded_sequences = encode_sequences(sequences, max_length=1000)

# 构建数据加载器
dataset = ProteinDataset(encoded_sequences, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 配置LSTM模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel(input_size=1, hidden_size=128, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# 训练LSTM模型
print("Training LSTM model...")
train_lstm_model(lstm_model, dataloader, criterion, optimizer, device, epochs=10)

# 提取特征并分类
print("Extracting LSTM features...")
lstm_features, lstm_labels = extract_lstm_features(lstm_model, dataloader, device)

print("Training Random Forest classifier...")
random_forest_classification(lstm_features, lstm_labels)


if __name__ == "__main__":
main()
```

仍然是因为数据量比较小，这个训练的结果也不好。综上来看，小数据量还是使用esm2系列模型最佳


### 2.参考文献：

Zhao, M., Lei, C., Zhou, K., Huang, Y., Fu, C., & Yang, S. (2024). POOE: predicting oomycete effectors based on a pre-trained large protein language model. mSystems. https://journals.asm.org/doi/abs/10.1128/msystems.01004-23