import argparse
import collections
import copy
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, get_activation_hook
from utils.kcenter_greedy import kCenterGreedy
from base import BaseDataLoader



# fix random seeds for reproducibility
# make sure this is fixed!
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    initial_data_loader = config.init_obj('data_loader', module_data)

    # build model architecture, then print to console
    initial_model = config.init_obj('arch', module_arch)
    model = copy.deepcopy(initial_model)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    initial_model = initial_model.to(device)
    model = model.to(device)
    if len(device_ids) > 1:
        initial_model = torch.nn.DataParallel(initial_model, device_ids=device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # Load original checkpoint
    
    checkpoint = torch.load(config.resume)
    initial_model.load_state_dict(checkpoint['state_dict'])

    #### Get the train idx using our method
    
    # Indices list
    all_train_idx = []
    # Activations list over train examples
    all_activations = []
    # Labels list over train examples
    all_targets = []
    
    # Budget of samples
    budget = config.budget
    
    # Method to store activations after forwards
    activations_dict = {}
    initial_model.linear.register_forward_hook(get_activation_hook("linear", activations_dict))
    
    initial_model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, index) in enumerate(initial_data_loader):
            data, target = data.to(device), target.to(device)

            output = initial_model(data)
            linear_activations = activations_dict["linear"]  # B x num_classes
            
            all_train_idx.append(index)
            all_activations.append(linear_activations)
            all_targets.append(target)
    
    activations_mat = torch.cat(all_activations, dim=0).cpu().numpy()  # Train set size x num_classes
    targets_mat = torch.cat(all_targets, dim=0).cpu().numpy()  # Train set size x num_classes
    
    # Intiailize method to get centers
    kcg = kCenterGreedy(activations_mat, targets_mat, SEED)
    selected_idx = kcg.select_batch_(model, [], budget)  # list of size budget
    # Index the train idx list properly
    selected_train_idx = torch.tensor(initial_data_loader.train_idx)[selected_idx]
    selected_train_idx = np.sort(selected_train_idx)
    np.savetxt(
        f"activation_cover_train_budget-{budget}_seed-{SEED}.csv",
        selected_train_idx,
        delimiter=", ",
        fmt="%d"
    )
    print("Saved activation cover!")
    
    # Data loader for selected subset of dataset
    data_loader = initial_data_loader.from_loader_and_data_subset(
        initial_data_loader,
        training=True,
        train_idx=selected_train_idx,
        return_index=False
    )
    valid_data_loader = data_loader.split_validation()
    # Remove resume from config so we start training from scratch
    config.resume = None
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler,
                                           optimizer)    
    
    # Train as normal
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    
    
# TODO:
# have a function to get a subsample of the dataset
# then rerun training from scratch (maybe with a new model)
# on the new dataset, and record perf/model separately


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to original checkpoint, to evaluate for dataset subset (default: None)')
    args.add_argument('-b', '--budget', type=int,
                      help='percentage of training set to keep in subset')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
