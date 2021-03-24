Sweeps ongoing:
* [Random search to understand which hyper-parameters matter](https://wandb.ai/wandb/xlsr/sweeps/1yovh01j)

## Instructions

* install requirements (requires `master` branch of `transformers)

  `pip install -r requirements.txt`

* make sure you're logged into W&B

  `wandb login`

* define your sweep configuration file (see `sweep.yaml`)

* create a sweep -> this will return a sweep id

  `wandb sweep sweep.yaml`

* launch an agent against the sweep (see the return from previous command)

  `wandb agent my_sweep_id`

  Notes:
  * you can launch as many agents as your machine handles, and use other machines. Just don't recreate another sweep and use the same sweep id.
  * if you want to add runs to my ongoing sweep (and didn't change this repo), you can launch `wandb agent wandb/xlsr/1yovh01j`
