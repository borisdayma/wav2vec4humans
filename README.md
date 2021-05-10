# Wav2Vec4Humans - Speech Recognition for Humans

*Transcribe audio without pronuncing the punctuation*

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

## Run on OVH

#### Optional: Build a Docker image

Note: you can just use my Docker image: [borisdayma/wav2vec4humans](https://hub.docker.com/r/borisdayma/wav2vec4humans)

To build your own Docker image:

```
$ docker build -t username/wav2vec4humans -f Dockerfile .
```

To push it to Docker Hub:

```
$ docker push username/wav2vec4humans
```

#### Launch OVH instance

Set up ovhai:
```
$ ovhai login
$ ovhai config set BHS `#choose BHS or GRA based on your region`
```

To launch an instance:
```
$ ovhai job run \
        --gpu 1 \
        -v datasets@BHS:/workspace/datasets:rw:cache `#pre-processed datasets` \
        -v cache@BHS:/workspace/.cache:rw:cache `#cache requires high capacity` \
        -e WANDB_API_KEY=xxxxx `#insert your key for auto-login` \
        borisdayma/wav2vec4humans `#you can choose your own docker image` \
```

Notes:

* once your dataset is created, you can load the volume in "ro" (read-only) instead of "rw" to avoid final sync
* you can automatically launch a command by adding `-- my_command`, for example `-- wandb agent my_sweep_id`
* remove local cache `rm -rf ~/.cache/**` before terminating your instance or it will take a very long time to sync back to its object storage (and each time you try to reload it)
* to remove cache storage in BHS region: `ovhai data delete -ay bhs cache && ovhai data delete -y bhs cache`

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

## Acknowledgements

This project would not have been possible without the help of so many, in particular:

* [W&B](http://docs.wandb.com/) for the great tracking & visualization tools for ML experiments ;
* [HuggingFace](https://huggingface.co/) for providing a great framework for Natural Language Understanding ;
* [wav2vec2-sprint](https://github.com/patil-suraj/wav2vec2-sprint) from Suraj Patil for helping me create the docker file ;
* [OVH cloud](https://www.ovh.com/) for the great cloud computing infrastructure ;
* the open source community who participated in the xlsr-wav2vec2 fine-tuning week and shared so many great tips!
