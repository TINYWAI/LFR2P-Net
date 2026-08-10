import os
import re
import pdb
import shutil
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []

        else:
            self.count = 0
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.all = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
            self.sum = np.sum(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count
            self.all = self.sum / 3600


class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, latest_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.latest_iter = latest_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.latest_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr'] * self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.latest_iter + 1
        self.latest_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
            param_group['lr'] = lr


class IterPolyLRScheduler(LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_iter: after this step, we stop decreasing learning rate
        min_lr: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_iter, min_lr, power=0.9, cur_iter=-1):
        if max_iter <= 1.:
            raise ValueError('max_iter should be greater than 1.')
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.power = power
        self.cur_iter = cur_iter
        super().__init__(optimizer)

    def get_lr(self):
        if self.cur_iter > self.max_iter:
            return [self.min_lr for _ in self.base_lrs]

        return [(base_lr - self.min_lr) *
                ((1 - self.cur_iter / self.max_iter) ** self.power) +
                self.min_lr for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.cur_iter + 1
        self.cur_iter = step if step != 0 else 1
        if self.cur_iter <= self.max_iter:
            decay_lrs = [(base_lr - self.min_lr) *
                         ((1 - self.cur_iter / self.max_iter) ** self.power) +
                         self.min_lr for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_confusion_matrix(save_path, state, step, matrix, class_names=None, normalize=True):
    """
    将混淆矩阵绘制成热力图并保存
    :param save_path: 图片保存路径 (如 'confusion_matrix.png')
    :param class_names: 类别名称列表 (如 ['Background', 'Damage'])
    :param normalize: 是否进行归一化（显示百分比）
    """
    _epsilon = 1e-7
    plt.figure(figsize=(10, 8))

    # matrix = self.confusion_matrix.astype('float')

    if normalize:
        # 每一行代表真实标签，除以每一行的和得到召回率方向的百分比
        matrix = matrix / (matrix.sum(axis=1)[:, np.newaxis] + _epsilon)
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Pixel Counts)'

    # 使用 Seaborn 绘制热力图
    ax = sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues',
                     annot_kws={"size": 15},
                     xticklabels=class_names,
                     yticklabels=class_names)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    # plt.title(title)
    ax.set_ylabel('True Label', fontsize=20)
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.tick_params(labelsize=15)

    if not os.path.exists(f'{save_path}/confusion_matrix'):
        os.makedirs(f'{save_path}/confusion_matrix')
    fig_path = f'{save_path}/confusion_matrix/{state}_{step}_confusion_matrix.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_state(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/checkpoints/iter_{}_checkpoint.pth.tar'.format(save_path, state['step'])
    latest_path = '{}/checkpoints/latest_checkpoint.pth.tar'.format(save_path)
    torch.save(state, model_path)
    shutil.copyfile(model_path, latest_path)


def save_state_single(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/checkpoints/iter_{}_checkpoint.pth.tar'.format(save_path, state['step'])
    torch.save(state, model_path)
    return model_path


def save_add_loss_state(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/checkpoints/iter_{}_{}_checkpoint.pth.tar'.format(save_path, state['step'], state['add_loss_name'])
    latest_path = '{}/checkpoints/latest_{}_checkpoint.pth.tar'.format(save_path, state['add_loss_name'])
    torch.save(state, model_path)
    shutil.copyfile(model_path, latest_path)


def load_add_loss_state(path, model):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=map_func)
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)
    model.load_state_dict(checkpoint['state_dict'])


def load_state(path, model, logger=None, latest_flag=True, optimizer=None):
    # pdb.set_trace()
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path) and latest_flag is False:
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' not in checkpoint and 'step' not in checkpoint:
            checkpoint = {'state_dict': checkpoint, 'step': -1}
    elif os.path.isfile(path) and latest_flag is True:
        checkpoint = torch.load(path, map_location='cpu')
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)
    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        if logger != None:
            logger.info('caution: missing keys from checkpoint {}: {}'.format(path, k))
        else:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

    copying_layers = {}
    ignoring_layers = {}
    for key in own_keys:
        if key not in ckpt_keys:
            continue
        if checkpoint['state_dict'][key].shape == model.state_dict()[key].shape:
            copying_layers[key] = checkpoint['state_dict'][key]
        else:
            ignoring_layers[key] = checkpoint['state_dict'][key]
            if logger != None:
                logger.info('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))
            else:
                print('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))

    model.load_state_dict(copying_layers, strict=False)
    eval_iteration = checkpoint['step']
    if logger != None:
        logger.info("=> loaded state from checkpoint '{}' (iter {})".format(path, eval_iteration))
    else:
        print("=> loaded state from checkpoint '{}' (iter {})".format(path, eval_iteration))
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        if logger != None:
            logger.info("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, eval_iteration))
        else:
            print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, eval_iteration))
    return eval_iteration


def load_imgnet_models(model, path, logger):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        state_dict = torch.load(path, map_location=map_func)
        if 'densenet' in path:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        mapped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # print(key)
            mapped_key = key
            mapped_state_dict[mapped_key] = value
            if 'running_var' in key:
                mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1)
        if 'vgg16' in path:
            mapped_state_dict['fc0.0.weight'] = mapped_state_dict['classifier.0.weight']
            mapped_state_dict['fc0.0.bias'] = mapped_state_dict['classifier.0.bias']
            mapped_state_dict['fc1.0.weight'] = mapped_state_dict['classifier.3.weight']
            mapped_state_dict['fc1.0.bias'] = mapped_state_dict['classifier.3.bias']
        elif 'alexnet' in path:
            mapped_state_dict['fc0.0.weight'] = mapped_state_dict['classifier.1.weight']
            mapped_state_dict['fc0.0.bias'] = mapped_state_dict['classifier.1.bias']
            mapped_state_dict['fc1.0.weight'] = mapped_state_dict['classifier.4.weight']
            mapped_state_dict['fc1.0.bias'] = mapped_state_dict['classifier.4.bias']

        state_dict = mapped_state_dict
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    # pdb.set_trace()
    for k in missing_keys:
        logger.info('caution: missing keys from checkpoint {}: {}'.format(path, k))

    model.load_state_dict(state_dict, False)


def param_groups(model):
    conv_weight_group = []
    conv_bias_group = []
    bn_group = []
    feature_weight_group = []
    feature_bias_group = []
    classification_fc_group = []

    normal_group = []
    arranged_names = set()

    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_group.append(m.weight)
            bn_group.append(m.bias)
            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')
        elif isinstance(m, nn.Conv2d):
            conv_weight_group.append(m.weight)
            if m.bias is not None:
                conv_bias_group.append(m.bias)
            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')
        elif isinstance(m, nn.Linear):
            if m.out_features == model.num_classes:
                classification_fc_group.append(m.weight)
                if m.bias is not None:
                    classification_fc_group.append(m.bias)
            else:
                feature_weight_group.append(m.weight)
                if m.bias is not None:
                    feature_bias_group.append(m.bias)

            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')

    for name, param in model.named_parameters():
        if name in arranged_names:
            continue
        else:
            normal_group.append(param)

    return conv_weight_group, conv_bias_group, bn_group, \
        feature_weight_group, feature_bias_group, classification_fc_group, \
        normal_group
