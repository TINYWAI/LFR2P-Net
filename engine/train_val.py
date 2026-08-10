import os
import sys

this_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
lib_path = os.path.join('../', this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import time
import yaml
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import prettytable as pt
from easydict import EasyDict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import core.models as models
import core.lossfunc as lossfunc
import datasets.datasets_catalog as dc
from core.utils.cd_dataset import BDADataset
from core.utils.sampler_utils import set_seed_, TrainIterationSampler, bda_collate
from core.utils.misc import create_dirs, create_logger, AverageMeter, save_state, load_state, IterLRScheduler, \
    IterPolyLRScheduler
from core.utils.bda_iou_metric import Evaluator

parser = argparse.ArgumentParser(description='PyTorch Change Detection Training')
parser.add_argument('--config', default='cfgs/bright_bs8_lfr2p-net_vmt.yaml')
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    file_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.TRAIN.CKPT = os.path.join(cfg.TRAIN.CKPT, file_name)

device_ids = cfg.TRAIN.DEVICE_IDS
torch.cuda.set_device(device_ids[0])
set_seed_(cfg.TRAIN.SEED)


class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """

    def __init__(self, cfg):
        """
        Initialize the Trainer with arguments from the command line or defaults.

        :param args: Argparse namespace containing:
            - dataset, train_dataset_path, holdout_dataset_path, etc.
            - model_type, model_param_path, resume path for checkpoint
            - learning rate, weight decay, etc.
        """
        self.cfg = cfg

        model = models.__dict__[cfg.MODEL.type](cfg=cfg.MODEL)
        logger.info(
            "=> creating model: \n{}".format(model))
        self.model = model.cuda()

        loc_criterion = lossfunc.__dict__[cfg.LOC_LOSS.MAIN_LOSS.TYPE](cfg=cfg.LOC_LOSS.MAIN_LOSS).cuda()
        clf_criterion = lossfunc.__dict__[cfg.CLF_LOSS.MAIN_LOSS.TYPE](cfg=cfg.CLF_LOSS.MAIN_LOSS).cuda()
        cm_criterion = lossfunc.__dict__[cfg.CM_LOSS.TYPE](cfg=cfg.CM_LOSS).cuda()

        extra_loc_criterion, extra_clf_criterion = None, None
        if 'EXTRA_LOSS' in cfg.LOC_LOSS:
            extra_loc_criterion = lossfunc.__dict__[cfg.LOC_LOSS.EXTRA_LOSS.TYPE](cfg=cfg.LOC_LOSS.EXTRA_LOSS).cuda()
        if 'EXTRA_LOSS' in cfg.CLF_LOSS:
            extra_clf_criterion = lossfunc.__dict__[cfg.CLF_LOSS.EXTRA_LOSS.TYPE](cfg=cfg.CLF_LOSS.EXTRA_LOSS).cuda()

        init_lr = cfg.SOLVER.BASE_LR
        optim_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                optim_params.append(p)
        for n, p in cm_criterion.named_parameters():
            if not p.requires_grad:
                continue
            else:
                optim_params.append(p)

        params = [
            {'params': optim_params},
        ]

        if cfg.SOLVER.OPTIM == 'SGD':
            optim = torch.optim.SGD(params, init_lr,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                    nesterov=cfg.SOLVER.get('NESTEROV', False))
        elif cfg.SOLVER.OPTIM == 'Adam':
            optim = torch.optim.Adam(params, init_lr,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        elif cfg.SOLVER.OPTIM == 'AdamW':
            optim = torch.optim.AdamW(params, init_lr,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise NotImplementedError()
        logger.info(optim)

        latest_iter = -1
        # optionally resume from a checkpoint
        if args.resume:
            if cfg.TRAIN.LOAD_PATH != '':
                logger.info('=> loading model: {}\n'.format(cfg.TRAIN.LOAD_PATH))
                latest_iter = load_state(cfg.TRAIN.LOAD_PATH, model, logger, latest_flag=False, optimizer=optim)
            elif cfg.TRAIN.LOAD_PATH == '':
                logger.info('=> loading latest saved model\n')
                latest_iter = load_state(cfg.TRAIN.CKPT, model, logger, latest_flag=True, optimizer=optim)
            else:
                assert True, 'wrong resume option'
        self.latest_iter = latest_iter

        model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
        self.model = model

        if cfg.SOLVER.TYPE == 'IterLRScheduler':
            lr_scheduler = IterLRScheduler(optim, cfg.SOLVER.LR_STEPS, cfg.SOLVER.LR_MULTS, latest_iter=latest_iter)
        elif cfg.SOLVER.TYPE == 'IterPolyLRScheduler':
            lr_scheduler = IterPolyLRScheduler(optim, cfg.SOLVER.MAX_ITER, cfg.SOLVER.MIN_LR,
                                               power=cfg.SOLVER.POWER,
                                               cur_iter=latest_iter)
        else:
            raise NotImplementedError()

        self.criterion = (loc_criterion, clf_criterion, cm_criterion, extra_loc_criterion, extra_clf_criterion)
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        train_dataset = BDADataset(cfg=cfg, mode='train')
        logger.info('train_set num: {}'.format(len(train_dataset)))
        if cfg.TRAIN.get('VAL') is True:
            val_dataset = BDADataset(cfg=cfg, mode='val')
            logger.info('val_set num: {}'.format(len(val_dataset)))
        else:
            val_dataset = None
        if cfg.TRAIN.get('TEST') is True:
            test_dataset = BDADataset(cfg=cfg, mode='test')
            logger.info('test_set num: {}'.format(len(test_dataset)))
        else:
            test_dataset = None

        train_sampler = TrainIterationSampler(dataset=train_dataset, total_iter=cfg.SOLVER.MAX_ITER,
                                              batch_size=train_dataset.batch_size, last_iter=self.latest_iter)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=train_dataset.batch_size,
            shuffle=False,
            num_workers=cfg.TRAIN.WORKERS, pin_memory=False, sampler=train_sampler)
        if cfg.TRAIN.get('VAL') is True:
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=val_dataset.batch_size, shuffle=False, drop_last=False,
                num_workers=cfg.TRAIN.WORKERS, pin_memory=False)
        else:
            val_loader = None
        if cfg.TRAIN.get('TEST') is True:
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=test_dataset.batch_size, shuffle=False, drop_last=False,
                num_workers=cfg.TRAIN.WORKERS, pin_memory=False)
        else:
            test_loader = None

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.clf_num_classes = train_dataset.da_num_classes

        self.train_loc = Evaluator(2)
        self.train_clf = Evaluator(self.clf_num_classes)
        self.train_total = Evaluator(self.clf_num_classes)
        self.eval_loc = Evaluator(2)
        self.eval_clf = Evaluator(self.clf_num_classes)
        self.eval_total = Evaluator(self.clf_num_classes)

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        cfg = self.cfg
        loc_criterion, clf_criterion, cm_criterion, extra_loc_criterion, extra_clf_criterion = self.criterion
        torch.cuda.empty_cache()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        f1_loc_metric = AverageMeter()
        f1_clf_metric = AverageMeter()
        f1_total_metric = AverageMeter()

        end = time.time()
        self.model.train()
        start_iter = self.latest_iter + 1
        for i, data in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            curr_step = start_iter + i
            self.lr_scheduler.step(curr_step)
            current_lr = self.lr_scheduler.get_lr()[0]

            pre_change_imgs, post_change_imgs, labels_loc, labels_clf = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            valid_labels_idx = (labels_clf != 255).any()
            if not valid_labels_idx: continue

            feat_dict, logits_dict = self.model(pre_change_imgs, post_change_imgs)
            loc_logits, clf_logits = logits_dict['loc_logits'], logits_dict['clf_logits']

            loc_main_loss = loc_criterion(feat_dict, loc_logits, labels_loc)
            clf_main_loss = clf_criterion(feat_dict, clf_logits, labels_clf)
            loc_main_loss, clf_main_loss = loc_main_loss.float(), clf_main_loss.float()
            loss = loc_main_loss + clf_main_loss

            extra_loc_loss, extra_clf_loss = None, None
            if extra_loc_loss is not None:
                loc_extra_loss = extra_loc_criterion(feat_dict, loc_logits, labels_loc)
                loc_extra_loss = loc_extra_loss.float()
                loss = loss + cfg.LOC_LOSS.EXTRA_LOSS.RATIO * loc_extra_loss
            if extra_clf_loss is not None:
                clf_extra_loss = extra_clf_criterion(feat_dict, clf_logits, labels_clf)
                clf_extra_loss = clf_extra_loss.float()
                loss = loss + cfg.CLF_LOSS.EXTRA_LOSS.RATIO * clf_extra_loss

            if cm_criterion is not None:
                cm_loss = cm_criterion(feat_dict, logits_dict, labels_loc, labels_clf)
                cm_loss = cm_loss.float()
                loss = loss + cm_loss

            # set gradient to zero
            self.optim.zero_grad()
            # backward
            loss.backward()
            # update params
            self.optim.step()

            loss = loss.float()
            losses.update(loss.item())

            if isinstance(loc_logits, (list, tuple)):
                loc_logits = loc_logits[0]
            if isinstance(clf_logits, (list, tuple)):
                clf_logits = clf_logits[0]

            output_loc = loc_logits.data.cpu().numpy()
            output_loc = np.argmax(output_loc, axis=1)
            labels_loc = labels_loc.cpu().numpy()

            output_clf = clf_logits.data.cpu().numpy()
            output_clf = np.argmax(output_clf, axis=1)
            labels_clf = labels_clf.cpu().numpy()

            self.train_loc.reset()
            self.train_clf.reset()
            self.train_total.reset()

            self.train_loc.add_batch(labels_loc, output_loc)
            output_clf_damage_part = output_clf[labels_loc > 0]
            labels_clf_damage_part = labels_clf[labels_loc > 0]
            self.train_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
            output_clf = output_clf * output_loc # filter background by loc map = 1
            self.train_total.add_batch(labels_clf, output_clf)

            loc_f1_score = self.train_loc.Pixel_F1_score()
            damage_f1_score = self.train_clf.Damage_F1_score()
            harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
            final_OA = self.train_total.Pixel_Accuracy()
            IoU_of_each_class = self.train_total.Intersection_over_Union()
            mIoU = self.train_total.Mean_Intersection_over_Union()
            total_f1_score = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

            f1_loc_metric.update(loc_f1_score)
            f1_clf_metric.update(harmonic_mean_f1)
            f1_total_metric.update(total_f1_score)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if curr_step % cfg.SOLVER.PRINT_FREQ == 0:
                tfb_logger.add_scalar('lr', current_lr, curr_step)
                tfb_logger.add_scalar('train/total_loss', losses.val, curr_step)
                tfb_logger.add_scalar('train/f1_loc', f1_loc_metric.val, curr_step)
                tfb_logger.add_scalar('train/f1_clf', f1_clf_metric.val, curr_step)
                tfb_logger.add_scalar('train/f1_total', f1_total_metric.val, curr_step)

                logger.info(f'Cfg: {os.path.basename(args.config)} |'
                            f'Iter: [{curr_step}/{len(self.train_loader) + start_iter}] |'
                            f'Time {batch_time.avg:.3f}({batch_time.val:.3f}) |'
                            f'Data {data_time.avg:.3f}({data_time.val:.3f}) |'
                            f'Total Loss {losses.avg:.4f}({losses.val:.4f}) |'
                            f'F1 Total Score {f1_total_metric.avg:.3f}({f1_total_metric.val:.3f}) |'
                            f'LOC_F1 {f1_loc_metric.avg:.3f}({f1_loc_metric.val:.3f}) |'
                            f'DA_F1 {f1_clf_metric.avg:.3f}({f1_clf_metric.val:.3f}) |'
                            f'LR {current_lr:.6f} |'
                            f'Total {batch_time.all:.2f}hrs |'
                            f'ETA {batch_time.avg / 3600 * (len(self.train_loader) - curr_step + 1):.2f}hrs |')

            if (curr_step + 1) % cfg.SOLVER.SNAPSHOT == 0:
                self.model.eval()

                save_state({
                    'step': curr_step + 1,
                    'dataset_name': cfg.TRAIN.DATASETS,
                    'type': cfg.MODEL.type,
                    'backbone': cfg.MODEL.get('encoder_name'),
                    'state_dict': self.model.module.state_dict(),
                    'aux_state_dict': cm_criterion.state_dict(),
                    'optimizer': self.optim.state_dict(),
                }, cfg.TRAIN.CKPT)

                self.eval(curr_step + 1, self.val_loader, 'val')
                # torch.cuda.empty_cache()
                self.eval(curr_step + 1, self.test_loader, 'test')
                # torch.cuda.empty_cache()
                self.model.train()

    def eval(self, eval_iteration, data_loader, state):
        logger.info('iter {}: {} ...'.format(eval_iteration, state))
        class_names = list(dc.get_vis_colors(cfg.TRAIN.DATASETS).keys())
        self.eval_loc.reset()
        self.eval_clf.reset()
        self.eval_total.reset()
        # torch.cuda.empty_cache()

        tqdm_num = len(data_loader.dataset)
        pbar = tqdm(total=tqdm_num)
        with torch.no_grad():
            for _, data in enumerate(data_loader):

                pre_change_imgs, post_change_imgs, labels_loc, labels_clf = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                feat_dict, logits_dict = self.model(pre_change_imgs, post_change_imgs)
                loc_logits, clf_logits = logits_dict['loc_logits'], logits_dict['clf_logits']
                if isinstance(loc_logits, (list, tuple)):
                    loc_logits = loc_logits[0]
                if isinstance(clf_logits, (list, tuple)):
                    clf_logits = clf_logits[0]

                output_loc = loc_logits.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = clf_logits.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.eval_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.eval_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                output_clf = output_clf * output_loc  # filter background by loc map = 1
                self.eval_total.add_batch(labels_clf, output_clf)

                pbar.update(pre_change_imgs.shape[0])
            pbar.close()

            loc_f1_score = self.eval_loc.Pixel_F1_score()
            damage_f1_score = self.eval_clf.Damage_F1_score()
            harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
            final_OA = self.eval_total.Pixel_Accuracy()
            IoU_of_each_class = self.eval_total.Intersection_over_Union()
            mIoU = self.eval_total.Mean_Intersection_over_Union()
            total_f1_score = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

            tb = pt.PrettyTable()
            tb.field_names = ['loc_f1', 'clf_f1', 'total_f1', 'final_OA', 'name', 'class', 'IoU']
            for i in range(self.clf_num_classes):
                if i == 0:
                    tb.add_row([loc_f1_score, harmonic_mean_f1, total_f1_score, final_OA, class_names[i], i,
                                IoU_of_each_class[i]])
                else:
                    tb.add_row(['', '', '', '', class_names[i], i, IoU_of_each_class[i]])
            tb.add_row(['', '', '', '', '', 'mean', mIoU])

            logger.info(tb)

            tfb_logger.add_scalar('{}/f1_loc'.format(state), loc_f1_score, eval_iteration)
            tfb_logger.add_scalar('{}/f1_clf'.format(state), harmonic_mean_f1, eval_iteration)
            tfb_logger.add_scalar('{}/f1_total'.format(state), total_f1_score, eval_iteration)
            tfb_logger.add_scalar('{}/final_OA'.format(state), final_OA, eval_iteration)
            tfb_logger.add_scalar('{}/mIoU'.format(state), mIoU, eval_iteration)


def main():
    create_dirs('{}/events'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/checkpoints'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/logs'.format(cfg.TRAIN.CKPT))
    global logger, tfb_logger
    if args.resume:
        logger = create_logger('global_logger', '{}/logs/log_resume.txt'.format(cfg.TRAIN.CKPT))
    else:
        logger = create_logger('global_logger', '{}/logs/log.txt'.format(cfg.TRAIN.CKPT))
    logger.info('{}'.format(cfg))
    tfb_logger = SummaryWriter('{}/events'.format(cfg.TRAIN.CKPT))
    shutil.copyfile(args.config, os.path.join(cfg.TRAIN.CKPT, args.config.split('/')[-1]))

    trainer = Trainer(cfg)
    trainer.training()


if __name__ == "__main__":
    main()
