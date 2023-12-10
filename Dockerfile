FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV ENV_DATASET=default
ENV ENV_CHILDDATASET=default
ENV ENV_RESULT=default
ENV ENV_NO=default
ENV ENV_NUM_EXAMPLES=200
ENV ENV_MODEL_BATCH_SIZE=32

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

## Install your dependencies here using apt install, etc.

RUN apt update && apt upgrade -y && apt clean
RUN apt install build-essential ffmpeg libsm6 libxext6 libsndfile1 libmagickwand-dev vim nano wget curl git -y

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN python -m pip install --upgrade pip

# list packages installed in the base image
RUN pip list

RUN mkdir -p /numbda-webnews-test

COPY ./requirements-no-torch.txt /numbda-webnews-test

WORKDIR /numbda-webnews-test

RUN pip install -r requirements-no-torch.txt

# list packages after installing requirements
RUN pip list

COPY ./ /numbda-webnews-test

# RUN python _download.py

# CMD ["python", "main.py"]
