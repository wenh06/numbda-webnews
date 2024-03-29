---
license: apache-2.0
language:
- zh
metrics:
- accuracy
pipeline_tag: text-classification
---
# Model Card for `numbda-webnews`

[中文版](README-zh.md)

<!-- Provide a quick summary of what the model is/does. -->

`numbda-webnews` is a news classification model fine-tuned from [roberta-base-finetuned-ifeng-chinese](https://huggingface.co/voidful/roberta-base-finetuned-ifeng-chinese) with a new dataset of approximately 40k news articles crawled from news websites in China, which is a sub-project of the [AI-Testing project](https://numbda.cs.tsinghua.edu.cn/AI-Testing/).

The dataset contains (not limited to) the following 14 categories:

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

The above 14 categories have a total of 26k samples.

## Model Details

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** <https://github.com/wenh06/numbda-webnews>
- **Huggingface Hub:** <https://huggingface.co/wenh06/numbda-webnews>

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# "wenh06/numbda-webnews" can be replaced with local path to the model directory
tokenizer = AutoTokenizer.from_pretrained("wenh06/numbda-webnews")
model = AutoModelForSequenceClassification.from_pretrained("wenh06/numbda-webnews")

pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
```

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

This model was fine-tuned using a new dataset of approximately 40k news articles crawled from news websites in China, which would be released latter some time.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

Evaluation results and software/hardware information can be found in [Weights & Biases](https://wandb.ai/wenh06/huggingface/runs/mg4uedxe/workspace?workspace=user-wenh06).

| Metric        | Score |
|---------------|-------|
| top1-accuracy | 0.768 |
| top3-accuracy | 0.944 |
| top5-accuracy | 0.981 |

### Curves of Top n Accuracy

| Top1 Accuracy | Top3 Accuracy | Top5 Accuracy |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="600" alt="eval-top1-acc.svg" src="images/eval-top1-acc.svg"> |  <img width="600" alt="eval-top3-acc.svg" src="images/eval-top3-acc.svg"> |  <img width="600" alt="eval-top5-acc.svg" src="images/eval-top5-acc.svg"> |
