Sweeps ongoing:
* [Random search to understand which hyper-parameters matter](https://wandb.ai/wandb/xlsr/sweeps/p23j88jo?workspace=user-borisd13)

## Instructions

* install requirements (requires `master` branch of `transformers)

  `pip install -r requirements.txt`

* make sure you're logged into W&B

  `wandb login`

* define your sweep configuration file (see `sweep.yaml`)

* create a sweep -> this will return a sweep id

  `wandb sweep sweep.yaml`

* launch an agent against the sweep

  `wandb agent my_sweep_id`
