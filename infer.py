import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

from customDataset import customDataset
from PIL import Image

logger = get_logger()

def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        # print(data)
        img = data['image']
        modal_x = data['depth']
        name = data['name']
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        # print(pred)
        # cv2.imwrite('./results/'+name+'.png', pred)
        # return
        # hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        # results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
        result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        class_colors = get_class_colors()
        palette_list = list(np.array(class_colors).flat)
        if len(palette_list) < 768:
            palette_list += [0] * (768 - len(palette_list))
        result_img.putpalette(palette_list)
        result_img.save('./results/'+name+'.png')

        # save raw result
        # cv2.imwrite('./results/'+name+'.png', pred)

        return
        # if self.save_path is not None:
        #     ensure_dir(self.save_path)
        #     ensure_dir(self.save_path+'_color')

        #     fn = name + '.png'

        #     # save colored result
        #     result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        #     class_colors = get_class_colors()
        #     palette_list = list(np.array(class_colors).flat)
        #     if len(palette_list) < 768:
        #         palette_list += [0] * (768 - len(palette_list))
        #     result_img.putpalette(palette_list)
        #     result_img.save(os.path.join(self.save_path+'_color', fn))

        #     # save raw result
        #     cv2.imwrite(os.path.join(self.save_path, fn), pred)
        #     logger.info('Save the image ' + fn)

        # if self.show_image:
        #     colors = self.dataset.get_class_colors
        #     image = img
        #     clean = np.zeros(label.shape)
        #     comp_img = show_img(colors, config.background, image, clean,
        #                         label,
        #                         pred)
        #     cv2.imshow('comp_image', comp_img)
        #     cv2.waitKey(0)

        # return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_pre = ValPre()
    # dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    pred_data = customDataset("../data/ScanNet/imgs/scene0005_01","../data/ScanNet/depths/scene0005_01")
    # pred_loader = DataLoader(pred_data,batch_size=int(args.batch_size))
    
    with torch.no_grad():
        segmentor = SegEvaluator(pred_data, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)