---
license: apache-2.0
language:
- zh
metrics:
- accuracy
pipeline_tag: text-classification
---
# Model Card for `numbda-webnews`

<!-- Provide a quick summary of what the model is/does. -->

`numbda-webnews` 是一个从 [roberta-base-finetuned-ifeng-chinese](https://huggingface.co/voidful/roberta-base-finetuned-ifeng-chinese) 微调得到的新闻分类模型。模型训练使用了一个新的数据集，该数据集包含了大约 4 万条从中国新闻网站爬取的新闻文章。这个数据集的构建是 [AI-Testing 项目](https://numbda.cs.tsinghua.edu.cn/AI-Testing/) 的一个子项目。

数据集包含（不限于）以下 14 个类别：

- 资讯
- 财经
- 体育
- 时政
- 娱乐
- 社会
- 科技
- 汽车
- 健康
- 萌宠
- 国际
- 生活
- 美食
- 游戏

以上 14 个类别共有 2.6 万条样本。

## 模型介绍

### 模型仓库

<!-- Provide the basic links for the model. -->

- **Repository:** <https://github.com/wenh06/numbda-webnews>
- **Huggingface Hub:** <https://huggingface.co/wenh06/numbda-webnews>

## 模型使用方法

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# "wenh06/numbda-webnews" can be replaced with local path to the model directory
tokenizer = AutoTokenizer.from_pretrained("wenh06/numbda-webnews")
model = AutoModelForSequenceClassification.from_pretrained("wenh06/numbda-webnews")

pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
```

## 模型训练

### 训练数据

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

本模型使用了一个新闻分类数据集，该数据集包含了大约 4 万条从中国新闻网站爬取的新闻文章。这个数据集的构建是 [AI-Testing 项目](https://numbda.cs.tsinghua.edu.cn/AI-Testing/) 的一个子项目，将在后续开源发布。

## 模型评测

<!-- This section describes the evaluation protocols and provides the results. -->

模型评测结果和软硬件信息可以在 [Weights & Biases](https://wandb.ai/wenh06/huggingface/runs/mg4uedxe/workspace?workspace=user-wenh06) 中找到。

| 评测指标       | 结果  |
|---------------|-------|
| top1-accuracy | 0.768 |
| top3-accuracy | 0.944 |
| top5-accuracy | 0.981 |

### Top n 准确率曲线

| Top1 Accuracy | Top3 Accuracy | Top5 Accuracy |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="600" alt="eval-top1-acc.svg" src="images/eval-top1-acc.svg"> |  <img width="600" alt="eval-top3-acc.svg" src="images/eval-top3-acc.svg"> |  <img width="600" alt="eval-top5-acc.svg" src="images/eval-top5-acc.svg"> |
