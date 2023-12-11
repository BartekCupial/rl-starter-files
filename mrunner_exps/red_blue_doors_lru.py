from mrunner.helpers.specification_helper import create_experiments_helper
from mrunner_exps.utils import combine_config_with_defaults

name = globals()["script"][:-3]

# params for all exps
config = {
    "algo": "ppo",
    "tag": name,
    "env": "MiniGrid-RedBlueDoors-6x6-v0",
    "save-interval": 10,
    "recurrence": 4,
    "frames": 1000000,
    "lru": True,
}
config = combine_config_with_defaults(config)

params_grid = {
    "seed": list(range(3)),
    "lru-layers": [3, 2, 1],
}


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="rl-starter-files",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
)
