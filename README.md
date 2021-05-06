# Wav2Vec4Humans - Speech Recognition for Humans

*Transcribe audio without having to pronounce punctuation*

## Introduction

I developed Wav2Vec4Humans because I didn't understand why we still had to talk like robots when speaking to our "smart" objects at the age of self-driving cars.

This project creates speech recognition models that also output punctuation so people can talk naturally.

It is based on finetuning a pre-trained [Wav2Vec2 model](https://arxiv.org/abs/2006.11477) using [HuggingFace](https://huggingface.co/).

## Try it!

The following models have been developped:

- TODO

In order to test it…

TODO add instructions

## How does it work?

To understand how the model was developed, check my W&B report. TODO add report.

## Usage

To train your own speech model:

* install requirements

  `pip install -r requirements.txt`

* make sure you're logged into W&B

  `wandb login`

* create a preprocessing function for your language

  TODO add more details

* run the training script

  TODO insert full command with comments on parameters.

You can also use [W&B sweeps](https://docs.wandb.ai/) to optimize hyper parameters:

* define your sweep configuration file

  update language in `sweep.yaml`

* create a sweep -> this will return a sweep id

  `wandb sweep sweep.yaml`

* launch an agent against the sweep

  `wandb agent my_sweep_id`

## About

*Built by Boris Dayma*

[![Follow](https://img.shields.io/twitter/follow/borisdayma?style=social)](https://twitter.com/intent/follow?screen_name=borisdayma)

For more details, visit the project repository.

[![GitHub stars](https://img.shields.io/github/stars/borisdayma/huggingtweets?style=social)](https://github.com/borisdayma/huggingtweets)

## Resources

* W&B report
* [HuggingFace and W&B integration documentation](https://docs.wandb.com/library/integrations/huggingface)

## Got questions about W&B?

If you have any questions about using W&B to track your model performance and predictions, please reach out to the [slack community](http://bit.ly/wandb-forum).
