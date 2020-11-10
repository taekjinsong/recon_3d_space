"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import cv2
import numpy as np
import os
import argparse

from visualize import ScannetVis
from tsdf_scan_whole import Scan
import fusion_whole

from collections import defaultdict
from data import cfg, set_cfg, set_dataset
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from yolact import Yolact
from utils.functions import SavePath

import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
from queue import Queue
from torch.multiprocessing import Process, set_start_method



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


# ======================================================================================================== #
# (Optional) This is an example of how to compute the 3D bounds
# in world coordinates of the convex hull of all camera view
# frustums in the dataset
# ======================================================================================================== #
use_yolact = True
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, help='ScanNet or handmade Dataset folder name (ex, scene0000, lab_kitchen17)',
                    default='lab_kitchen17')
parser.add_argument('--is_scannet', dest='is_scannet', default=False, action='store_true', help='True if it is scannet dataset')
parser.add_argument('--data_root', dest='data_root', default='/home/tjsong/data/', type=str, help='Data root directory')
parser.add_argument('--skip_imgs', dest='skip_imgs', default=10, type=int, help='number of skipped image while reconstruct 3D space')
parser.add_argument('--start_img', dest='start_img', default=0, type=int, help='index of start image for reconstruction')

if use_yolact:
    parser.add_argument('--trained_model',
                        default='weights/yolact_plus_resnet50_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

global args
args = parser.parse_args()


data_folder = os.path.join(args.data_root, args.scene)
# data_folder = '/media/data_root/data/' + args.scene
intrinsic_folder = 'intrinsic'
intrinsic_file = 'intrinsic_depth.txt'
root_path = os.getcwd()

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

print("Estimating voxel volume bounds...")
skip_imgs = args.skip_imgs
start_imgs = args.start_img
cam_intr = np.loadtxt(os.path.join(data_folder, intrinsic_folder, intrinsic_file), delimiter=' ')
cam_intr = cam_intr[:3, :3]
mesh_plot = True
use_gpu = True
scannet_data = args.is_scannet

def get_file_names(path):
  files = os.listdir(path)
  files = sorted(files, key=lambda x: int(x.split('.')[0]))
  file_names = [path + fn for fn in files]

  return file_names

if (args.scene.split('_')[0] == 'realsense'):
    rgb_paths = os.path.join(data_folder, 'rgb/')
else:
    rgb_paths = os.path.join(data_folder, 'color/')
rgb_names = get_file_names(rgb_paths)

depth_paths = os.path.join(data_folder, 'depth/')
depth_names = get_file_names(depth_paths)

pose_paths = os.path.join(data_folder, 'pose/')
pose_names = get_file_names(pose_paths)


if args.config is not None:
    set_cfg(args.config)

if args.trained_model == 'interrupt':
    args.trained_model = SavePath.get_interrupt('weights/')
elif args.trained_model == 'latest':
    args.trained_model = SavePath.get_latest('weights/', cfg.name)

if args.config is None:
    model_path = SavePath.from_str(args.trained_model)
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

if args.detect:
    cfg.eval_mask_branch = False

if args.dataset is not None:
    set_dataset(args.dataset)

with torch.no_grad():
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.resume and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_map(ap_data)
        exit()

    dataset = None

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    print(' Done.')

    if args.cuda:
        net = net.cuda()

    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug



scan = Scan(rgb_paths=rgb_paths, depth_paths=depth_paths, pose_paths=pose_paths,
            cam_intr=cam_intr, mesh_plot=mesh_plot, scannet_data=scannet_data, mask_net=net,
            args=args, root_path=root_path, use_gpu=use_gpu)

app = QApplication(sys.argv)

vis = ScannetVis(scan=scan,
                 rgb_names=rgb_names,
                 depth_names=depth_names,
                 pose_names=pose_names,
                 offset=start_imgs,
                 skip_im=skip_imgs,
                 mesh_plot=mesh_plot,
                 parent=None
                 )


# set_start_method('spawn', force=True)

# print instructions
print("To navigate:")
print("\tn: next (next scan)")
print("\tq: quit (exit program)")

print("To control 3D view:")
print("\tLMB: orbits the view around its center point")
print("\tRMB or scroll: change scale_factor (i.e. zoom level)")
print("\tSHIFT + LMB: translate the center point")
print("\tSHIFT + RMB: change FOV")

# p1 = Process(target=update_queue, args=(vis,))
# p2 = Process(target=start_vis, args=(vis,))
# p1.start()
# p2.start()
# p1.join()
# p2.join()

vis.run()
vis.show()

sys.exit(app.exec_())



