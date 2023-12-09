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

`numbda-webnews` is a news classification model fine-tuned from [roberta-base-finetuned-ifeng-chinese](https://huggingface.co/voidful/roberta-base-finetuned-ifeng-chinese) with a new dataset of approximately 40k news articles crawled from news websites in China.

The dataset contains the following 14 categories:

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

## Model Details

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** <https://github.com/wenh06/numbda-webnews>

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

This model was fine-tuned using a new dataset of approximately 40k news articles crawled from news websites in China, which would be released latter some time.
