import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn.functional as F
import os


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 hyper_tune=False,
                 record_el2n=False, at_epochs=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, hyper_tune)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.record_el2n = record_el2n
        self.at_epochs = at_epochs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        if self.record_el2n and self.at_epochs is not None and (self.at_epochs == True or epoch in self.at_epochs):
            all_data_idx = []
            all_el2n_score = []
            el2n = True
        else:
            el2n = False

        for batch_idx, data_info in enumerate(self.data_loader):
            if len(data_info) == 3:
                data, target, data_idx = data_info
            elif len(data_info) == 2:
                data, target = data_info
            else:
                raise ValueError(f"Expecting data_info from data_loader to have size 3, got: {len(data_info)}")

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if el2n:
                with torch.no_grad():
                    normalized_output = F.softmax(output, dim=-1)
                    el2n_score = torch.linalg.norm(normalized_output - F.one_hot(target, num_classes=normalized_output.size(-1)).to(output.dtype), dim=1)
                    all_data_idx.append(data_idx)
                    all_el2n_score.append(el2n_score)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation or self.hyper_tune:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if el2n:
            with torch.no_grad():
                all_el2n_score = torch.cat(all_el2n_score)
                all_data_idx = torch.cat(all_data_idx)
                el2n_sorted, el2n_sort_indices = torch.sort(all_el2n_score, descending=True)
                data_idx_sorted_with_el2n = all_data_idx[el2n_sort_indices]
                el2n_to_save = np.stack((data_idx_sorted_with_el2n.cpu().detach().numpy(), el2n_sorted.cpu().detach().numpy()))
                os.makedirs(self.config.el2n_dir, exist_ok=True)
                with open(os.path.join(self.config.el2n_dir, f'el2n_epoch{epoch}.npy'), 'wb') as f:
                    np.save(f, el2n_to_save)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data_info in enumerate(self.valid_data_loader):
                if len(data_info) == 3:
                    data, target, data_idx = data_info
                elif len(data_info) == 2:
                    data, target = data_info
                else:
                    raise ValueError(f"Expecting data_info from data_loader to have size 3, got: {len(data_info)}")

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
