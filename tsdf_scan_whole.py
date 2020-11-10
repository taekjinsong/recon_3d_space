import numpy as np

from collections import Counter
import cv2
import os
import fusion_whole

import torch
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from data import cfg, set_cfg, set_dataset, COLORS
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
from collections import defaultdict

class Scan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, rgb_paths=None, depth_paths=None, pose_paths=None, cam_intr=None, mesh_plot=True,
               scannet_data=True, mask_net=None, args=None, root_path=None, use_gpu=True):
    self.rgb_paths = rgb_paths
    self.depth_paths = depth_paths
    self.pose_paths = pose_paths
    self.cam_intr = cam_intr
    self.vol_bnds = np.zeros((3, 2))
    self.prev_vol_bnds = np.zeros((3, 2))
    self.vol_bnds_created = False
    self.count_update = 0
    self.mesh_plot = mesh_plot
    self.scannet_data = scannet_data
    self.mask_net = mask_net
    self.args = args
    self.color_cache = defaultdict(lambda: {})
    self.ROOT_PATH = root_path
    self.use_gpu = use_gpu

    self.prev_color_image = None
    self.prev_masks = None
    self.prev_num_dets_to_consider = 0
    self.prev_text_str = None
    self.masked_img_number = 0

    # self.reset()

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, rgb_names, depth_names, pose_names, frame_num, recon=True):
    """ Open raw scan and fill in attributes
    """
    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(rgb_names), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_names, -1).astype(float)
    self.depth_im = depth_im
    if self.scannet_data:
      depth_im /= 1000.
    else:
      depth_im /= 5000.    # axus action pro
    # depth_im[depth_im == 65.535] = 0
    depth_im[depth_im >= 3.5] = 0
    # depth_im[depth_im == 100.535] = 0
    cam_pose = np.loadtxt(pose_names)  # 4x4 rigid transformation matrix
    self.cam_pose = cam_pose

    # apply YOLACT to get mask info
    img_H, img_W = depth_im.shape
    color_image = cv2.resize(color_image, (img_W, img_H), interpolation=cv2.INTER_AREA)

    self.color_im = color_image
    self.depth_im = depth_im
    frame = torch.from_numpy(color_image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = self.mask_net(batch)
    self.preds = preds

    self.masks, self.masks_color, self.text_str = None, None, None
    img_numpy = self.prep_display(preds, frame, None, None, undo_transform=False)
    self.masked_img = img_numpy

    if (recon):
        # compute camera view frustum and extend convex hull
        # transform current view to 3D global view
        view_frust_pts = fusion_whole.get_view_frustum(depth_im, self.cam_intr, cam_pose)
        self.view_frust_pts = view_frust_pts
        # find minimum and maximum (x,y,z) value in global view
        self.vol_min = self.vol_bnds[:, 0] - np.amin(view_frust_pts, axis=1)
        self.vol_max = np.amax(view_frust_pts, axis=1) - self.vol_bnds[:, 1]

        self.vol_bnds[:, 0] = np.minimum(self.vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        self.vol_bnds[:, 1] = np.maximum(self.vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

        if (self.vol_bnds_created):
          self.tsdf_vol.update(self.vol_bnds, self.prev_vol_bnds, self.vol_min, self.vol_max)
          self.prev_vol_bnds = np.copy(self.vol_bnds)
          self.count_update += 1
        else:
          print("Initializing voxel volume...")
          # voxel_size : 0.04
          self.tsdf_vol = fusion_whole.TSDFVolume(self.vol_bnds, voxel_size=0.08, use_gpu=self.use_gpu,
                                                  root_path=self.ROOT_PATH, cfg=cfg)
          self.prev_vol_bnds = np.copy(self.vol_bnds)
          self.vol_bnds_created = True


        # ======================================================================================================== #
        # Integrate
        # ======================================================================================================== #
        # Integrate voxel volume
        if (self.num_dets_to_consider > 0):
            self.prev_color_image = color_image.copy()
            self.prev_masks = self.masks.clone()
            self.prev_num_dets_to_consider = self.num_dets_to_consider
            self.prev_text_str = self.text_str
            self.masked_img_number += 1

        if (self.masked_img_number <= 1):
            first_masking = 1
        else:
            first_masking = 0
        self.tsdf_vol.integrate(color_image, depth_im, self.cam_intr, cam_pose,
                                      self.boxes,
                                      self.masks, self.masks_color, self.num_dets_to_consider, self.text_str,
                                      self.prev_color_image, self.prev_masks, self.prev_num_dets_to_consider,
                                      self.prev_text_str, first_masking, frame_num,
                                      obs_weight=1.)


        if self.mesh_plot:
          self.verts, self.faces, self.norms, self.colors = self.tsdf_vol.get_mesh()
          return self.verts, self.faces, self.norms, self.colors
        else:
          self.point_cloud = self.tsdf_vol.get_point_cloud()
          return self.point_cloud
    else:
        return None, None, None, None

  def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = self.args.display_lincomb,
                                        crop_masks        = self.args.crop,
                                        score_threshold   = self.args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:self.args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        self.classes, self.scores, self.boxes = classes, scores, boxes

    num_dets_to_consider = min(self.args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < self.args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
            return self.color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                self.color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if self.args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # remove overlapped area of mask results
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            overlapped_list = []
            box_size = int((x2-x1)*(y2-y1))
            color = get_color(j)
            score = scores[j]
            for k in reversed(range(num_dets_to_consider)):
                if (k != j):
                    a1, b1, a2, b2 = boxes[k, :]
                    box_size_sub = int((a2-a1)*(b2-b1))
                    if ((min(a2, x2) - max(a1, x1) > 0) and (min(b2, y2) - max(b1, y1) > 0)):
                        # overlapped area
                        S_jk = (min(a2, x2) - max(a1, x1)) * (min(b2, y2) - max(b1, y1))
                        if (S_jk / box_size > 0.9):
                            # included other BBox
                            pass
                        elif (S_jk / box_size_sub > 0.3):
                            # Subtract overlapped area in current bounding box
                            # Find overlapped Bbox position
                            x_list = [x1, x2, a1, a2]
                            y_list = [y1, y2, b1, b2]
                            x_list.sort()
                            y_list.sort()
                            overlapped_list.append([int(x_list[1]), int(y_list[1]), int(x_list[2]), int(y_list[2])])
            for ov_bbox in overlapped_list:
                masks[j][ov_bbox[1]: ov_bbox[3], ov_bbox[0]: ov_bbox[2]] = 0

        self.masks = masks

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        self.masks_color = colors
        self.masks_color_2 = masks_color

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        self.img_gpu = img_gpu

    if self.args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if self.args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    self.num_dets_to_consider = num_dets_to_consider
    if num_dets_to_consider == 0:
        return img_numpy

    if self.args.display_text or self.args.display_bboxes:
        self.text_str = {}
        draw_masks = self.masks.squeeze(-1).to(torch.device("cpu")).detach().numpy().astype(np.float32)
        update_masks = self.masks.clone()
        overlapped_list = []
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            box_size = int((x2-x1)*(y2-y1))
            color = get_color(j)
            score = scores[j]

            if self.args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if self.args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s:%d_%.2f' % (_class, classes[j], score) if self.args.display_scores else _class
                self.text_str[j] = text_str
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 + 15)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 + text_h + 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_numpy


