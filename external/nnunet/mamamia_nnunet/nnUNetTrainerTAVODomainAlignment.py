import math
import os
from os import replace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTAVOSaveEveryEpoch import nnUNetTrainerTAVOSaveEveryEpoch
TARGET_PREFIXES = {'NACT': 'NACT_', 'ISPY1': 'ISPY1_', 'DUKE': 'DUKE_', 'ISPY2': 'ISPY2_'}

class GradientReversalFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.alpha, None)

def grl(x, alpha):
    return GradientReversalFn.apply(x, alpha)

class PooledMapDiscriminator(nn.Module):

    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        if x.ndim > 2:
            x = x.mean(dim=tuple(range(2, x.ndim)))
        return self.net(x)

def main_output(output):
    return output[0] if isinstance(output, (list, tuple)) else output

def pixel_entropy_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1, keepdim=True)

def pooled_probabilities(logits):
    return F.softmax(logits, dim=1).mean(dim=tuple(range(2, logits.ndim)))

def rbf_mmd(x, y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    loss = x.new_tensor(0.0)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)
        loss = loss + torch.exp(-gamma * xx).mean()
        loss = loss + torch.exp(-gamma * yy).mean()
        loss = loss - 2.0 * torch.exp(-gamma * xy).mean()
    return loss / float(len(sigmas))

class nnUNetTrainerTAVOAlignmentBase(nnUNetTrainerTAVOSaveEveryEpoch):
    alignment_method = 'none'
    uses_discriminator = False
    discriminator_input = 'probabilities'

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.target_domain = os.environ.get('MAMAMIA_TARGET', '').upper()
        self.target_prefix = TARGET_PREFIXES.get(self.target_domain, '')
        self.alignment_weight = float(os.environ.get('ADAPT_LAMBDA', '0.1'))
        self.domain_weight = float(os.environ.get('ADAPT_DOMAIN_LAMBDA', str(self.alignment_weight)))
        self.entropy_weight = float(os.environ.get('ADAPT_ENTROPY_LAMBDA', str(self.alignment_weight)))
        self.semantic_weight = float(os.environ.get('ADAPT_SEMANTIC_LAMBDA', str(self.alignment_weight)))
        self.discriminator_hidden = int(os.environ.get('ADAPT_DISC_HIDDEN', '64'))
        if 'NNUNET_NUM_EPOCHS' in os.environ:
            self.num_epochs = int(os.environ['NNUNET_NUM_EPOCHS'])
        if 'NNUNET_ITERATIONS_PER_EPOCH' in os.environ:
            self.num_iterations_per_epoch = int(os.environ['NNUNET_ITERATIONS_PER_EPOCH'])
        if 'NNUNET_VAL_ITERATIONS_PER_EPOCH' in os.environ:
            self.num_val_iterations_per_epoch = int(os.environ['NNUNET_VAL_ITERATIONS_PER_EPOCH'])
        if 'NNUNET_BATCH_SIZE' in os.environ:
            self.batch_size = int(os.environ['NNUNET_BATCH_SIZE'])
        self.discriminator = None

    def configure_optimizers(self):
        if self.uses_discriminator:
            self.discriminator = PooledMapDiscriminator(self.label_manager.num_segmentation_heads, self.discriminator_hidden).to(self.device)
            params = list(self.network.parameters()) + list(self.discriminator.parameters())
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
            return (optimizer, lr_scheduler)
        return super().configure_optimizers()

    def _infer_target_prefix(self):
        if self.target_prefix:
            return self.target_prefix
        dataset_name = self.plans_manager.dataset_name.upper()
        for target, prefix in TARGET_PREFIXES.items():
            if f'MAMAMIA_{target}_' in dataset_name:
                self.target_domain = target
                self.target_prefix = prefix
                return prefix
        return ''

    def _domain_masks(self, keys, device):
        prefix = self._infer_target_prefix()
        is_target = torch.tensor([str(k).startswith(prefix) for k in keys], device=device, dtype=torch.bool)
        return (~is_target, is_target)

    def _domain_alpha(self):
        progress = float(self.current_epoch) / max(float(self.num_epochs - 1), 1.0)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def _disc_input(self, logits):
        if self.discriminator_input == 'entropy':
            ent = pixel_entropy_from_logits(logits)
            return ent.repeat(1, logits.shape[1], *[1] * (logits.ndim - 2))
        return F.softmax(logits, dim=1)

    def _alignment_loss(self, logits, source_mask, target_mask):
        return logits.new_tensor(0.0)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch.get('keys', [])
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            seg_loss = self.loss(output, target)
            logits = main_output(output)
            source_mask, target_mask = self._domain_masks(keys, logits.device)
            can_align = bool(source_mask.any() and target_mask.any())
            align_loss = self._alignment_loss(logits, source_mask, target_mask) if can_align else logits.new_tensor(0.0)
            loss = seg_loss + align_loss
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + (list(self.discriminator.parameters()) if self.discriminator is not None else []), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + (list(self.discriminator.parameters()) if self.discriminator is not None else []), 12)
            self.optimizer.step()
        return {'loss': loss.detach().cpu().numpy(), 'seg_loss': seg_loss.detach().cpu().numpy(), 'align_loss': align_loss.detach().cpu().numpy(), 'aligned_batches': np.array(float(source_mask.any() and target_mask.any()), dtype=np.float32)}

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        align_loss = float(np.mean(outputs.get('align_loss', [0.0])))
        seg_loss = float(np.mean(outputs.get('seg_loss', [0.0])))
        aligned = float(np.mean(outputs.get('aligned_batches', [0.0])))
        self.logger.my_fantastic_logging.setdefault('alignment_losses', [])
        self.logger.my_fantastic_logging.setdefault('segmentation_losses', [])
        self.logger.log('alignment_losses', align_loss, self.current_epoch)
        self.logger.log('segmentation_losses', seg_loss, self.current_epoch)
        self.print_to_log_file(f'{self.alignment_method}_seg_loss {np.round(seg_loss, decimals=4)} {self.alignment_method}_align_loss {np.round(align_loss, decimals=4)} aligned_batch_fraction {np.round(aligned, decimals=4)}')

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank != 0 or self.discriminator is None:
            return
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        checkpoint['alignment_method'] = self.alignment_method
        checkpoint['discriminator_state'] = self.discriminator.state_dict()
        tmp_filename = f'{filename}.tmp'
        torch.save(checkpoint, tmp_filename)
        replace(tmp_filename, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        super().load_checkpoint(filename_or_checkpoint)
        checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False) if isinstance(filename_or_checkpoint, str) else filename_or_checkpoint
        if self.discriminator is not None and 'discriminator_state' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])

class nnUNetTrainerTAVODANN(nnUNetTrainerTAVOAlignmentBase):
    alignment_method = 'dann'
    uses_discriminator = True
    discriminator_input = 'probabilities'

    def _alignment_loss(self, logits, source_mask, target_mask):
        x = self._disc_input(logits)
        alpha = self._domain_alpha()
        domain_logits = self.discriminator(grl(x, alpha))
        labels = torch.zeros_like(domain_logits)
        labels[source_mask] = 1.0
        return self.domain_weight * F.binary_cross_entropy_with_logits(domain_logits, labels)

class nnUNetTrainerTAVOMMD(nnUNetTrainerTAVOAlignmentBase):
    alignment_method = 'mmd'

    def _alignment_loss(self, logits, source_mask, target_mask):
        pooled = pooled_probabilities(logits)
        return self.alignment_weight * rbf_mmd(pooled[source_mask], pooled[target_mask])

class nnUNetTrainerTAVOADVENT(nnUNetTrainerTAVOAlignmentBase):
    alignment_method = 'advent'
    uses_discriminator = True
    discriminator_input = 'entropy'

    def _alignment_loss(self, logits, source_mask, target_mask):
        x = self._disc_input(logits)
        alpha = self._domain_alpha()
        domain_logits = self.discriminator(grl(x, alpha))
        labels = torch.zeros_like(domain_logits)
        labels[source_mask] = 1.0
        return self.domain_weight * F.binary_cross_entropy_with_logits(domain_logits, labels)

class nnUNetTrainerTAVOSEASA(nnUNetTrainerTAVOAlignmentBase):
    alignment_method = 'seasa'

    def _alignment_loss(self, logits, source_mask, target_mask):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs[target_mask] * log_probs[target_mask]).sum(dim=1).mean()
        source_sem = pooled_probabilities(logits[source_mask]).mean(dim=0)
        target_sem = pooled_probabilities(logits[target_mask]).mean(dim=0)
        semantic = F.mse_loss(target_sem, source_sem.detach())
        return self.entropy_weight * entropy + self.semantic_weight * semantic
