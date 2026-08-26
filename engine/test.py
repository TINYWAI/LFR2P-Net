import os
import sys
import threading
from skimage import io

this_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
lib_path = os.path.join('../', this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
import argparse
import time
import yaml
import logging
import numpy as np
from tqdm import tqdm
import prettytable as pt
from easydict import EasyDict

import torch
import torch.optim
import torch.nn.parallel
from torch.utils.data import DataLoader

import core.models as models
import datasets.datasets_catalog as dc
from core.utils.cd_dataset import BDADataset
from core.utils.sampler_utils import set_seed_
from core.utils.misc import create_dirs, create_logger, AverageMeter, load_state, save_confusion_matrix
from core.utils.bda_iou_metric import Evaluator, compute_disaster_event_metrics, compute_disaster_type_metrics

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--config', required=True, help='path to the YAML config file')
parser.add_argument('--ckpt', required=True, help='path to the trained checkpoint')

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    file_name = os.path.splitext(os.path.basename(args.config))[0]

device_ids = cfg.TEST.DEVICE_IDS
class_names = list(dc.get_vis_colors(cfg.TRAIN.DATASETS).keys())
torch.cuda.set_device(device_ids[0])
set_seed_(cfg.TRAIN.SEED)

thread_max_num = threading.Semaphore(4)


def label_map_color(masked_pred, cls_color_map=dc.get_vis_colors(cfg.TRAIN.DATASETS)):
    cm = np.array(list(cls_color_map.values())).astype(np.uint8)
    color_img = cm[masked_pred]
    return color_img


def get_disaster_events():
    """Returns a list of disaster events based on filename prefixes."""

    return ['bata-explosion', 'beirut-explosion', 'congo-volcano', 'haiti-earthquake', 'hawaii-wildfire',
            'la_palma-volcano', 'libya-flood', 'marshall-wildfire', 'mexico-hurricane', 'morocco-earthquake',
            'myanmar-hurricane', 'noto-earthquake', 'turkey-earthquake', 'ukraine-conflict']


def get_disaster_types():
    """Returns a list of disaster events based on filename prefixes."""
    return [
        "earthquake", "wildfire", "volcano", "explosion", "flood",
        "conflict", "hurricane"
    ]


def post_process_work(param):
    # save_dir, file_name, eval_iteration, state, output, target = param
    save_dir, file_name, eval_iteration, state, output_loc, output_clf, labels_loc, labels_clf = param
    num_classes = len(class_names)
    patch_loc = Evaluator(2)
    patch_clf = Evaluator(num_classes)
    patch_total = Evaluator(num_classes)

    patch_loc.add_batch(labels_loc, output_loc)
    output_clf_damage_part = output_clf[labels_loc > 0]
    labels_clf_damage_part = labels_clf[labels_loc > 0]
    patch_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
    patch_total.add_batch(labels_clf, output_clf)

    loc_f1_score = patch_loc.Pixel_F1_score()
    damage_f1_score = patch_clf.Damage_F1_score()
    harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
    final_OA = patch_total.Pixel_Accuracy()
    IoU_of_each_class = patch_total.Intersection_over_Union()
    mIoU = patch_total.Mean_Intersection_over_Union()
    total_f1_score = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

    with open('{}/patch_results_{}_{}.csv'.format(save_dir, state, eval_iteration), 'a') as fout:
        patch_msg = '{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            file_name, loc_f1_score, harmonic_mean_f1, total_f1_score, final_OA, mIoU)
        patch_msg = patch_msg + ', \r\n'
        fout.write(patch_msg)

    loc_pred = output_loc.astype(np.uint8)
    clf_pred = output_clf.astype(np.uint8)
    clf_pred = loc_pred * clf_pred

    clf_vis = label_map_color(clf_pred)
    save_path = '{}/pred_{}/{}'.format(save_dir, eval_iteration, file_name[:-len('tif')] + 'png')
    save_color_path = '{}/pred_color_{}/{}'.format(save_dir, eval_iteration, file_name[:-len('tif')] + 'png')
    if cfg.TEST.SAVE_PRED is True:
        io.imsave(save_path, clf_pred, check_contrast=False)
    if cfg.TEST.SAVE_VIS is True:
        io.imsave(save_color_path, clf_vis, check_contrast=False)


def main():
    state = cfg.TEST.DATALIST
    save_root = 'results'
    log_dir = f'{save_root}/logs/'
    save_dir = f'{save_root}/{state}_results/'

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    logger = create_logger('global_logger', f'{log_dir}/{state}_log.txt')
    logger.info('{}'.format(cfg))

    model = models.__dict__[cfg.MODEL.type](cfg=cfg.MODEL)
    logger.info("=> creating model: \n{}".format(model))

    test_dataset = BDADataset(cfg=cfg, mode=state)
    logger.info('test_set num: {}'.format(len(test_dataset)))

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_dataset.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.TRAIN.WORKERS, pin_memory=False)

    logger.info('=> loading model: {}\n'.format(args.ckpt))
    eval_iteration = load_state(args.ckpt, model, latest_flag=False, optimizer=None)

    logger = logging.getLogger('global_logger')
    data_time = AverageMeter()

    clf_num_classes = len(class_names)
    test_loc = Evaluator(2)
    test_clf = Evaluator(clf_num_classes)
    test_total = Evaluator(clf_num_classes)

    disaster_type_evaluator_dict = {event: Evaluator(num_class=clf_num_classes) for event in get_disaster_types()}
    disaster_event_evaluator_dict = {event: Evaluator(num_class=clf_num_classes) for event in get_disaster_events()}

    data_loader = test_loader
    logger.info('iter {}: {} ...'.format(eval_iteration, state))
    if cfg.TEST.SAVE_PRED is True:
        create_dirs('{}/pred_{}'.format(save_dir, eval_iteration))
    if cfg.TEST.SAVE_VIS is True:
        create_dirs('{}/pred_color_{}'.format(save_dir, eval_iteration))
    with open('{}/patch_results_{}_{}.csv'.format(save_dir, state, eval_iteration), 'a') as fout:
        patch_msg = 'file_name, loc_f1_score, harmonic_mean_f1, total_f1_score, final_OA, mIoU, \r\n'
        fout.write(patch_msg)

    model = model.cuda()
    model.eval()
    end = time.time()
    pbar = tqdm(total=len(test_dataset))

    index = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            pre_change_imgs, post_change_imgs, labels_loc, labels_clf = data
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            feat_dict, logits_dict = model(pre_change_imgs, post_change_imgs)
            loc_logits, clf_logits = logits_dict['loc_logits'], logits_dict['clf_logits']
            if isinstance(loc_logits, (list, tuple)):
                loc_logits = loc_logits[0]
            if isinstance(clf_logits, (list, tuple)):
                clf_logits = clf_logits[0]

            labels_loc = labels_loc.cpu().numpy()
            labels_clf = labels_clf.cpu().numpy()
            output_loc, output_clf = get_pred(loc_logits, clf_logits)

            with thread_max_num:
                thread_list = []
                current_batch_size = pre_change_imgs.shape[0]
                for j in range(current_batch_size):
                    file_name = os.path.basename(test_dataset.metas[index + j][2])
                    # pred = pred_maps[j]
                    labels_loc_idx = labels_loc[j, ...]
                    labels_clf_idx = labels_clf[j, ...]
                    output_loc_idx = output_loc[j, ...]
                    output_clf_idx = output_clf[j, ...]
                    params = (save_dir, file_name, eval_iteration, state, output_loc_idx, output_clf_idx,
                              labels_loc_idx, labels_clf_idx)
                    thread = threading.Thread(target=post_process_work, args=(params,))
                    thread.start()
                    thread_list.append(thread)
                for thread in thread_list:
                    thread.join()

            test_loc.add_batch(labels_loc, output_loc)
            output_clf_damage_part = output_clf[labels_loc > 0]
            labels_clf_damage_part = labels_clf[labels_loc > 0]
            test_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
            output_clf = output_clf * output_loc
            test_total.add_batch(labels_clf, output_clf)

            for disaster_type in disaster_type_evaluator_dict.keys():
                if disaster_type in file_name:
                    disaster_type_evaluator_dict[disaster_type].add_batch(labels_clf, output_clf)
                    break  # Only match one event

            for event in disaster_event_evaluator_dict.keys():
                if event in file_name:
                    disaster_event_evaluator_dict[event].add_batch(labels_clf, output_clf)
                    break  # Only match one event

            index += current_batch_size
            pbar.update(current_batch_size)
        pbar.close()

        loc_f1_score = test_loc.Pixel_F1_score()
        damage_f1_score = test_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        final_OA = test_total.Pixel_Accuracy()
        IoU_of_each_class = test_total.Intersection_over_Union()
        mIoU = test_total.Mean_Intersection_over_Union()
        total_f1_score = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

        tb = pt.PrettyTable()
        tb.field_names = ['loc_f1', 'clf_f1', 'total_f1', 'final_OA', 'name', 'class', 'IoU']
        for i in range(clf_num_classes):
            if i == 0:
                tb.add_row([loc_f1_score, harmonic_mean_f1, total_f1_score, final_OA, class_names[i], i,
                            IoU_of_each_class[i]])
            else:
                tb.add_row(['', '', '', '', class_names[i], i, IoU_of_each_class[i]])
        tb.add_row(['', '', '', '', '', 'mean', mIoU])
        logger.info(tb)

        event_list, miou_list, iou_list, average_mIoU = compute_disaster_event_metrics(disaster_event_evaluator_dict)
        tb = pt.PrettyTable()
        tb.field_names = ['event', 'miou', 'background', 'intact', 'damaged', 'destroyed']
        for i in range(len(event_list)):
            tb.add_row([event_list[i], miou_list[i], *list(iou_list[i])])
        tb.add_row(['average_mIoU', average_mIoU, '', '', '', ''])
        logger.info(tb)

        disaster_list, miou_list, iou_list, average_mIoU = compute_disaster_type_metrics(disaster_type_evaluator_dict)
        tb = pt.PrettyTable()
        tb.field_names = ['disaster_type', 'miou', 'background', 'intact', 'damaged', 'destroyed']
        for i in range(len(disaster_list)):
            tb.add_row([disaster_list[i], miou_list[i], *list(iou_list[i])])
        tb.add_row(['average_mIoU', average_mIoU, '', '', '', ''])
        logger.info(tb)

        matrix = test_total.confusion_matrix.astype('float')
        save_confusion_matrix(save_dir, state, eval_iteration, matrix, class_names=class_names,
                              normalize=True)


def get_pred(loc_logits, clf_logits):
    output_loc = loc_logits.data.cpu().numpy()
    output_loc = np.argmax(output_loc, axis=1)

    output_clf = clf_logits.data.cpu().numpy()
    output_clf = np.argmax(output_clf, axis=1)

    return output_loc, output_clf


if __name__ == '__main__':
    main()
