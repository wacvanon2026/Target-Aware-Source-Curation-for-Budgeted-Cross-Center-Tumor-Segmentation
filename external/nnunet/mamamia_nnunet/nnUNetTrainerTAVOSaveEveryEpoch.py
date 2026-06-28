from os import replace
from os.path import join
import numpy as np
import torch
from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerTAVOSaveEveryEpoch(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, unpack_dataset=unpack_dataset, device=device)
        self.save_every = 1
        self.num_epochs = 50

    def on_epoch_end(self):
        from time import time
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")
        current_epoch = self.current_epoch
        self.save_checkpoint(join(self.output_folder, f'checkpoint_epoch_{current_epoch:03d}.pth'))
        self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f'New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}')
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file('No checkpoint written, checkpointing is disabled')
            return
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        checkpoint = {'network_weights': mod.state_dict(), 'optimizer_state': self.optimizer.state_dict(), 'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None, 'logging': self.logger.get_checkpoint(), '_best_ema': self._best_ema, 'current_epoch': self.current_epoch + 1, 'init_args': self.my_init_kwargs, 'trainer_name': self.__class__.__name__, 'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes}
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state'] = self.lr_scheduler.state_dict()
        tmp_filename = f'{filename}.tmp'
        torch.save(checkpoint, tmp_filename)
        replace(tmp_filename, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        super().load_checkpoint(filename_or_checkpoint)
        checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False) if isinstance(filename_or_checkpoint, str) else filename_or_checkpoint
        if self.lr_scheduler is not None and 'lr_scheduler_state' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
