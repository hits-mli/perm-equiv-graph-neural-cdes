import os
import sys
import yaml
import logging
import json

import wandb

# Add the grandparent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, grandparent_dir)


logger = logging.getLogger("wandb_sweep")
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    with open("configs/dynamical_systems/oversampling/astgcn_config.yaml", "r") as file:
        config_yaml = yaml.safe_load(file)

    if config_yaml["model"]["name"] in ["astgcn", "dcrnn", "stid_gcn"]:
        from engine.trainer_oversampling_recurrent import Trainer
    else:
        from engine.trainer_oversampling import Trainer

    logger.info(json.dumps(config_yaml, indent=4))

    with wandb.init(config=config_yaml, project=config_yaml["wandb"]["project"]) as run:
        trainer = Trainer(**config_yaml)
        best_validation_loss = trainer.run()
