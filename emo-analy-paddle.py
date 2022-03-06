#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[5]:


get_ipython().system('unzip -oq /home/aistudio/data/data53469/COTE-DP.zip -d work/')
get_ipython().system('unzip -oq /home/aistudio/data/data53469/COTE-BD.zip -d work/')
get_ipython().system('unzip -oq /home/aistudio/data/data53469/NLPCC14-SC.zip -d work/')
get_ipython().system('unzip -oq /home/aistudio/data/data53469/COTE-MFW.zip -d work/')
get_ipython().system('unzip -oq /home/aistudio/data/data53469/ChnSentiCorp.zip -d work/')
get_ipython().system('unzip -oq /home/aistudio/data/data53469/SE-ABSA16_PHNS.zip -d work/')
get_ipython().system('unzip -oq /home/aistudio/data/data53469/SE-ABSA16_CAME.zip -d work/')


# In[2]:


get_ipython().system(' tree work/ -d')


# In[3]:


import cv2
import matplotlib.pyplot as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

pyplot.imshow(cv2.imread("work/WechatIMG17.jpeg"))


# 语义分割数据集抽样可视化

# In[ ]:


import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
get_ipython().run_line_magic('matplotlib', 'inline')

image_path_list = ['work/WechatIMG17.jpeg', 'samples/images/H0005.jpg']
label_path_list = [path.replace('images', 'labels').replace('jpg', 'png')
                   for path in image_path_list]

plt.figure(figsize=(8, 8))
for i in range(len(image_path_list)):
    plt.subplot(len(image_path_list), 2, i*2+1)
    plt.title(image_path_list[i])
    plt.imshow(cv2.imread(image_path_list[i])[:, :, ::-1])

    plt.subplot(len(image_path_list), 2, i*2+2)
    plt.title(label_path_list[i])
    plt.imshow(cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE))
plt.tight_layout()
plt.show()


# 数据切分 数据集类的定义

# In[1]:


import paddle
import numpy as np
import paddle.vision.transforms as T


class MyImageNetDataset(paddle.io.Dataset):
    def __init__(self,
                 num_samples,
                 num_classes):
        super(MyImageNetDataset, self).__init__()

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=127.5, std=127.5)])

    def __getitem__(self, index):
        image = np.random.randint(low=0, high=256, size=(512, 512, 3))
        label = np.random.randint(low=0, high=self.num_classes, size=(1,))

        image = image.astype('float32')
        label = label.astype('int64')

        image = self.transform(image)

        return image, label

    def __len__(self):
        return self.num_samples


# In[2]:


# 计算图像数据整体均值和方差
import glob
import numpy as np


def get_mean_std(image_path_list):
    print('Total images:', len(image_path_list))
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        for c in range(3):
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    mean /= len(image_path_list)
    std /= len(image_path_list)

    mean /= max_val - min_val
    std /= max_val - min_val

    return mean, std


# 数据集类的测试

# In[14]:


train_dataset = MyImageNetDataset(num_samples=1200, num_classes=1000)
print(len(train_dataset))

image, label = train_dataset[0]
print(image.shape, label.shape)


for image, label in train_dataset:
    print(image.shape, label.shape)
    break


# In[16]:


train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=False)

for step, data in enumerate(train_dataloader):
    image, label = data
    print(step, image.shape, label.shape)


# 定义数据处理方法，将训练数据进行分布式批处理，其他数据进行批处理
# 同时定义了个计算损失函数和路由函数的方法

# In[8]:


import numpy as np
import paddle


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu


# In[11]:


# 加载自定义数据集，自定义划分验证集。
from paddle.io import Dataset, Subset
from paddlenlp.datasets import MapDataset

class BaseDateset(Dataset):
    def __init__(self, data, is_test = False):
        self._data = data
        self._is_test = is_test
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        example = {}
        samples = self._data[idx].split('\t')
        if self._is_test:
            qid = samples[-2]
            label = ''
            text = samples[-1]
        else:
            qid = ''
            label = int(samples[-2])
            text = samples[-1]
            
        example['text'] = text
        example['label'] = label
        example['qid'] = qid
        return example

def open_func(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f.readlines()[1:]:
            if len(line.strip().split('\t')) >= 2:
                samples.append(line.strip())
    return samples

data = open_func('work/COTE-BD/train.tsv')
baseset = BaseDateset(data)
sub_train_ds = Subset(dataset=baseset, indices=[i for i in range(len(baseset)) if i % 5 !=4])
sub_dev_ds = Subset(dataset=baseset, indices=[i for i in range(len(baseset)) if i % 5 ==4])
train_ds = MapDataset(sub_train_ds)
dev_ds = MapDataset(sub_dev_ds)

data = open_func('work/COTE-BD/test.tsv')
baseset = BaseDateset(data, is_test=True)
test_ds = MapDataset(baseset)


# In[2]:


get_ipython().system('pip install --upgrade paddlenlp -i https://pypi.org/simple')


# 模型使用的是：百度正式发布情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。

# In[3]:


from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch", num_classes=2)
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")


# In[4]:


import os
from functools import partial
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
   
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid


# In[12]:


# 批量数据大小
batch_size = 32
# 文本序列最大长度
max_seq_length = 128

# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)


# In[15]:


import time
import paddlenlp

# 训练轮次
epochs = 8
# 训练过程中保存模型参数的文件夹
ckpt_dir = "skep_ckpt"
# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

print(num_training_steps, int(num_training_steps*0.1))

scheduler = paddlenlp.transformers.LinearDecayWithWarmup(5E-5, num_training_steps, 0.1)
# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=scheduler,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()


# In[14]:


# 开启训练
global_step = 0
tic_train = time.time()
best_accu = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s, lr: %.7f"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train), scheduler.get_lr()))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        scheduler.step()
        optimizer.step()
        optimizer.clear_grad()

    save_dir = os.path.join(ckpt_dir, "model_%d" % epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 评估当前训练的模型
    accu = evaluate(model, criterion, metric, dev_data_loader)
    # 保存当前模型参数等
    model.save_pretrained(save_dir)
    # 保存tokenizer的词表等
    tokenizer.save_pretrained(save_dir)
    if float(accu) > best_accu:
        save_dir = os.path.join(ckpt_dir, "model_best")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存当前模型参数等
        model.save_pretrained(save_dir)
        # 保存tokenizer的词表等
        tokenizer.save_pretrained(save_dir)
        best_accu = accu
    print('best_accu:', best_accu)


# In[6]:


import numpy as np
import paddle

# 处理测试集数据
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack() # qid
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)


# In[ ]:


# 根据实际运行情况，更换加载的参数路径
params_path = 'skep_ckp/model_best/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)


# In[ ]:


label_map = {0: '0', 1: '1'}
results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))


# In[ ]:


res_dir = "./results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 写入预测结果
with open(os.path.join(res_dir, "NLPCC14-SC4.tsv"), 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for qid, label in results:
        f.write(str(qid[0])+"\t"+label+"\n")


# In[ ]:





# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
