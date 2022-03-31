import argparse
import torch
import numpy as np
from parse_config import ConfigParser
import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from parse_config import ConfigParser
from functools import partial
from train import main

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

ray.init(num_gpus=1)

def hyper_tune(train_config, tune_config, num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    work_dir = os.getcwd()
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", 'val_loss', 'val_accuracy', "training_iteration"])
    result = tune.run(
        partial(single_job, train_config=train_config, ori_work_dir=work_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(work_dir, "hyper_results"))
    best_trial = result.get_best_trial("val_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_accuracy"]))


def single_job(tune_config, train_config, ori_work_dir=None):
    train_config._config["trainer"]["save_dir"] = os.getcwd()
    if ori_work_dir is not None:
        os.chdir(ori_work_dir)
    train_config = ConfigParser(train_config._config, modification=tune_config)
    main(train_config, hyper_tune=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)

    tune_config = {
        "optimizer;args;lr": tune.loguniform(1e-4, 1e-1),
        "data_loader;args;batch_size": tune.choice([32, 64, 128, 256]),
    }
    hyper_tune(config, tune_config, num_samples=10, max_num_epochs=10, gpus_per_trial=1)
