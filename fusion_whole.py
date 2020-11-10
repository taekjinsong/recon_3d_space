# Copyright (c) 2018 Andy Zeng

import numpy as np
import torch

from numba import njit, prange
from skimage import measure
import matplotlib.cm as cm
from graphviz import Digraph
import webcolors
import os
import cv2

# For color clustering
import operator
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from stitch import Stitch
from skimage.measure import ransac, EllipseModel

# from data import COLORS
import warnings
warnings.filterwarnings(action='ignore')


try:
  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule
  FUSION_GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  FUSION_GPU_MODE = 0


class Get_Color_Info(object):
  def __init__(self):
    self.COLORS = \
      ((244,  67,  54),
       (233,  30,  99),
       (156,  39, 176),
       (103,  58, 183),
       ( 63,  81, 181),
       ( 33, 150, 243),
       (  3, 169, 244),
       (  0, 188, 212),
       (  0, 150, 136),
       ( 76, 175,  80),
       (139, 195,  74),
       (205, 220,  57),
       (255, 235,  59),
       (255, 193,   7),
       (255, 152,   0),
       (255,  87,  34),
       (121,  85,  72),
       (158, 158, 158),
       ( 96, 125, 139))

  def closest_colour(self, requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
      r_c, g_c, b_c = webcolors.hex_to_rgb(key)
      rd = (r_c - requested_colour[0]) ** 2
      gd = (g_c - requested_colour[1]) ** 2
      bd = (b_c - requested_colour[2]) ** 2
      min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

  def get_colour_name(self, requested_colour):
    try:
      closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
      closest_name = self.closest_colour(requested_colour)
      actual_name = None
    return actual_name, closest_name

  def RGB2HEX(self, color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

  def get_color_hist_kmeans(self, color_list):
    clf = KMeans(n_clusters=4)
    labels = clf.fit_predict(color_list)
    counts = Counter(labels)
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i[0]] for i in sorted_counts]
    hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    color_hist = []
    for i, rgb in enumerate(rgb_colors):
      actual_name, closest_name = self.get_colour_name(list(np.int_(rgb)))
      if (actual_name == None):
        color_hist.append([hex_colors[i], closest_name])
      else:
        color_hist.append([hex_colors[i], actual_name])

    return color_hist


class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size, use_gpu=True, root_path=None, cfg=None):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    self.cfg = cfg

    # setting root path
    self.root_path = os.path.join(root_path, 'scene_results')
    if not os.path.exists(self.root_path):
      os.mkdir(self.root_path)
    self.f_idx = 0
    # setting save info into folders (scene_graph, bounidng_box)
    self.bbox_path = os.path.join(self.root_path, 'BBOX')
    if not os.path.exists(self.bbox_path):
      os.mkdir(self.bbox_path)
    self.scene_graph_path = os.path.join(self.root_path, 'scene_graph')
    if not os.path.exists(self.scene_graph_path):
      os.mkdir(self.scene_graph_path)

    # Initialize color system
    self.GCI = Get_Color_Info()
    colors = cm.rainbow(np.linspace(0, 1, 80))
    self.class_colors = (colors*255)[:,:3].astype('int')

    # Initialize scene graph data
    self.node_data = {}
    self.rel_data = {}

    # Initialize same node detection system
    self.debug_same_node_detector = True
    self.ID_2D = 0
    self.stitch_img = None
    self.wide_class_mask = None
    self.apply_panorama = False
    self.stitch_method = Stitch()
    self.stitch_list = {}
    self.thumbnail_dict = {}


    ''' GFTT-BRIEF '''
    self.feature_detector = cv2.GFTTDetector_create(
      maxCorners=1000, minDistance=12.0,
      qualityLevel=0.001, useHarrisDetector=False)
    self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
      bytes=32, use_orientation=False)
    self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # setting several techniques
    self.is_depth_clustered = True
    ''' cluster technique : ['kmeans', 'meanshift', 'ransac', 'dbscan', 'hdbscan']'''
    self.cluster_technique = 'ransac'
    self.cluster_object = ['dining table', 'bench', 'bottle', 'cup', 'chair', 'bed', 'couch']
    self.mask_data = np.ones([])

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 3*self._voxel_size  # truncation on SDF (orig default : 5)
    self._trunc_margin_mask = 2*self._voxel_size
    self._color_const = 256 * 256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)
    self._prev_vol_bnds = self._vol_bnds.copy()


    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._class_vol_cpu = -np.ones(self._vol_dim).astype(np.int32)

    self.gpu_mode = use_gpu and FUSION_GPU_MODE
    self._prev_vol_bnds = np.copy(self._vol_bnds)


    # Copy voxel volumes to GPU
    if self.gpu_mode:
      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)
      self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
      cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)

      # Cuda kernel function (C++)
      self._cuda_src_mod = SourceModule("""
        #include <stdio.h>
        #include <math.h>
       
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  int * class_vol,
                                  int * mask_data,
                                  float * mask_color_data,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int num_mask = (int) other_params[6];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
              
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          
                    
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
              
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float trunc_margin_mask = other_params[7];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
              
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
                    
          int update_tsdf = (int) other_params[9];
          int first_masking = (int) other_params[10];
          
          if (update_tsdf == 1){
            weight_vol[voxel_idx] = w_new;
            tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;         
          }
          
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          //new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          //new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          //new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          if (color_vol[voxel_idx] == 0.0 && class_vol[voxel_idx] == -1 && update_tsdf == 1)
            color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
            
          
          // compare Yolact masked points with current image pixels
          // mask_data : [cls_score][cls_label][2D_instance]
          // class_vol is composed [detect_cnt][cls_score][cls_label][3D_instance]
          int is_first_masked_img = (int) other_params[8];
          if (is_first_masked_img == 0)
            return;
          
          int pixel_xy = pixel_y * im_w + pixel_x;
          if (num_mask == 0 or mask_data[pixel_xy] == -1)
            return;
            
          float mask_depth_diff = cam_pt_z - depth_value;
          if (mask_depth_diff < -trunc_margin_mask)
            return;
          
          int prev_class_vol = class_vol[voxel_idx];
          int prev_mask_data = mask_data[pixel_xy];
          int curr_cls_score = (int) (prev_mask_data/10000);
          int curr_cls_label = ((int)(prev_mask_data/100)) - ((int)(prev_mask_data/10000))*100;
          int instance_2D_ID = ((int) prev_mask_data) - ((int)(prev_mask_data/100))*100;
          
          int detected_cnt_num = 1;
          if (first_masking) {
            class_vol[voxel_idx] = mask_data[pixel_xy] + 1000000;
          }
          else{
            int new_id = 1;
            int voxel_neigh_idx = 0;
            float neigh_class = -1.0;
            // dining table : 60, bed: 59, couch : 57, chair:56, refrigerator : 72, bicycle : 1
            int neigh_gap = 3;
            if (curr_cls_label == 60 || curr_cls_label == 59 || curr_cls_label == 57 || curr_cls_label == 72)
              neigh_gap = 10;
            if (curr_cls_label == 56 || curr_cls_label == 1)
              neigh_gap = 6;
            for (int i = -neigh_gap; i < neigh_gap; i++){
              for (int j = -neigh_gap; j < neigh_gap; j ++){
                for (int k = -neigh_gap; k < neigh_gap; k ++){
                  if (voxel_x+i >= 0 && voxel_y+j >= 0 && voxel_z+k >= 0 && voxel_x+i <= vol_dim_x && voxel_y+j <= vol_dim_y && voxel_z+k <= vol_dim_z){
                    voxel_neigh_idx = ((int)voxel_x + i)*vol_dim_y*vol_dim_z + ((int)voxel_y + j)*vol_dim_z + ((int)voxel_z + k);
                    neigh_class = class_vol[voxel_neigh_idx];
                    if (neigh_class != -1.0){
                      int prev_detected_cnt = (int) (neigh_class/1000000);
                      int prev_cls_score = ((int)(neigh_class/10000)) - ((int)(neigh_class/1000000))*100;
                      int prev_cls_label = ((int)(neigh_class/100)) - ((int)(neigh_class/10000))*100;
                      int prev_instance_3D_ID = ((int) prev_class_vol) - ((int)(neigh_class/100))*100;
                      if (prev_cls_label == curr_cls_label){
                        class_vol[voxel_idx] = neigh_class;
                        new_id = 0;
                      }
                    }
                  }
                }
              }
            }
            if (new_id){
              // update new_id
              class_vol[voxel_idx] = mask_data[pixel_xy] + 1000000;
            }
          }
          
          if (detected_cnt_num >= 1){
            new_color = - (mask_color_data[3*curr_cls_label + 2] + mask_color_data[3*curr_cls_label + 1]*256 + mask_color_data[3*curr_cls_label + 0]*256*256);
            new_b = floorf(new_color/(256*256));
            new_g = floorf((new_color-new_b*256*256)/256);
            new_r = new_color-new_b*256*256-new_g*256;
            color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
          }
        }
        
        """)

      self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

      # Determine block/grid size on GPU
      gpu_dev = cuda.Device(0)
      self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
      self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))


  def update(self, vol_bnds, prev_vol_bnds, vol_min, vol_max):
    # Update voxel volume parameters
    #self._prev_vol_bnds = prev_vol_bnds
    self._vol_bnds = self._prev_vol_bnds.copy()

    self._tsdf_vol_cpu, self._color_vol_cpu = self.get_volume()
    cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
    cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
    self._tsdf_vol_gpu.free()
    self._color_vol_gpu.free()
    self._weight_vol_gpu.free()
    self._class_vol_gpu.free()

    x_extend_b, y_extend_b, z_extend_b = (vol_min/self._voxel_size)
    x_extend_u, y_extend_u, z_extend_u = (vol_max/self._voxel_size)

    self.test1 = (x_extend_b, y_extend_b, z_extend_b)
    self.test2 = (x_extend_u, y_extend_u, z_extend_u)

    if (x_extend_b > 0):
      x_extend_b = np.ceil(x_extend_b).copy(order='C').astype(int)
      self._vol_bnds[:, 0] = self._prev_vol_bnds[:, 0] - np.array([x_extend_b*self._voxel_size, 0., 0.])
      self._vol_origin[0] = self._prev_vol_bnds[:, 0][0] - x_extend_b*self._voxel_size
      self._vol_bnds = np.add(self._vol_bnds, np.array([x_extend_b * self._voxel_size, 0., 0.]).reshape(3, 1))
      self._vol_dim += np.array([x_extend_b, 0, 0])
      for i in range(x_extend_b):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, 0, 1, axis=0)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, 0, 0, axis=0)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, 0, 0, axis=0)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, 0, -1, axis=0)
    if (y_extend_b > 0):
      y_extend_b = np.ceil(y_extend_b).copy(order='C').astype(int)
      self._vol_bnds[:, 0] = self._prev_vol_bnds[:, 0] - np.array([0., y_extend_b * self._voxel_size, 0.])
      self._vol_origin[1] = self._prev_vol_bnds[:, 0][1] - y_extend_b * self._voxel_size
      self._vol_bnds = np.add(self._vol_bnds, np.array([0., y_extend_b*self._voxel_size, 0.]).reshape(3, 1))
      self._vol_dim += np.array([0, y_extend_b, 0])
      for j in range(y_extend_b):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, 0, 1, axis=1)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, 0, 0, axis=1)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, 0, 0, axis=1)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, 0, -1, axis=1)
    if (z_extend_b > 0):
      z_extend_b = np.ceil(z_extend_b).copy(order='C').astype(int)
      self._vol_bnds[:, 0] = self._prev_vol_bnds[:, 0] - np.array([0., 0., z_extend_b * self._voxel_size])
      self._vol_origin[2] = self._prev_vol_bnds[:, 0][2] - z_extend_b * self._voxel_size
      #self._vol_bnds = np.add(self._vol_bnds, np.array([0., 0., z_extend_b*self._voxel_size]).reshape(3, 1))
      self._vol_dim += np.array([0, 0, z_extend_b])
      for k in range(z_extend_b):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, 0, 1, axis=2)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, 0, 0, axis=2)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, 0, 0, axis=2)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, 0, -1, axis=2)

    if (x_extend_u > 0):
      x_extend_u = np.ceil(x_extend_u).copy(order='C').astype(int)
      self._vol_bnds[:, 1] += np.array([x_extend_u, 0, 0])
      self._vol_dim += np.array([x_extend_u, 0, 0])
      for i in range(x_extend_u):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, self._tsdf_vol_cpu.shape[0], 1, axis=0)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, self._weight_vol_cpu.shape[0], 0, axis=0)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, self._color_vol_cpu.shape[0], 0, axis=0)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, self._class_vol_cpu.shape[0], -1, axis=0)
    if (y_extend_u > 0):
      y_extend_u = np.ceil(y_extend_u).copy(order='C').astype(int)
      self._vol_bnds[:, 1] += np.array([0, y_extend_u, 0])
      self._vol_dim += np.array([0, y_extend_u, 0])
      for j in range(y_extend_u):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, self._tsdf_vol_cpu.shape[1], 1, axis=1)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, self._weight_vol_cpu.shape[1], 0, axis=1)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, self._color_vol_cpu.shape[1], 0, axis=1)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, self._class_vol_cpu.shape[1], -1, axis=1)
    if (z_extend_u > 0):
      z_extend_u = np.ceil(z_extend_u).copy(order='C').astype(int)
      self._vol_bnds[:, 1] += np.array([0, 0, z_extend_u])
      self._vol_dim += np.array([0, 0, z_extend_u])
      for k in range(z_extend_u):
        self._tsdf_vol_cpu = np.insert(self._tsdf_vol_cpu, self._tsdf_vol_cpu.shape[2], 1, axis=2)
        self._weight_vol_cpu = np.insert(self._weight_vol_cpu, self._weight_vol_cpu.shape[2], 0, axis=2)
        self._color_vol_cpu = np.insert(self._color_vol_cpu, self._color_vol_cpu.shape[2], 0, axis=2)
        self._class_vol_cpu = np.insert(self._class_vol_cpu, self._class_vol_cpu.shape[2], -1, axis=2)
    self.move_pose = self._vol_origin - self._vol_bnds[:, 0]
    self._vol_bnds = np.add(self._vol_bnds, self.move_pose.reshape(3, 1))
    self._prev_vol_bnds = self._vol_bnds.copy()


    # Copy voxel volumes to GPU
    self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
    cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
    self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
    cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
    self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
    cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)
    self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
    cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)

    gpu_dev = cuda.Device(0)
    n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
    grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
    grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
    grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
    self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
    self._n_gpu_loops = int(
      np.ceil(float(np.prod(self._vol_dim)) / float(np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))


  def integrate(self, color_im, depth_im, cam_intr, cam_pose,
                boxes,
                masks, masks_color, num_masks, class_info,
                prev_color_im, prev_masks, prev_num_dets_to_consider, prev_text_str, first_masking, frame_num,
                obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """
    self.im_h, self.im_w = depth_im.shape
    self.cam_intr, self.cam_pose = cam_intr, cam_pose
    self.color_im = color_im.copy()
    self.masked_img = color_im.copy()

    self.prev_color_im = prev_color_im
    self.good_matches = []

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

    self.class_info = class_info
    comp_ID = []
    if num_masks > 0:
      self.masks_test = masks.squeeze(-1).to(torch.device("cpu")).detach().numpy().astype(np.float32)
      self.mask_data = -np.ones(self.masks_test[0].shape, dtype="int32").reshape(-1)
      self.seen_class = {}
      for i in range(num_masks):
        class_value = self.class_info[i]
        class_name = class_value.split(':')[0]
        class_index = int(class_value.split(':')[1].split('_')[0])
        class_score = int(float(class_value.split(':')[1].split('_')[1])*100)-1   # set value to 0 ~ 99
        self.class_score = class_score

        # add clustering method to depth --> get more precision masks
        if (self.is_depth_clustered):
          self.depth_val = np.zeros(self.masks_test[i].reshape(-1).shape)
          self.depth_val[self.masks_test[i].reshape(-1).nonzero()] = depth_im[self.masks_test[i].nonzero()]
          self.check_depth = self.depth_val[self.depth_val.nonzero()].reshape(-1, 1)

          if (self.cluster_technique == 'kmeans'):
            self.cluster = KMeans(n_clusters=3, random_state=0).fit(self.check_depth)

          if (self.cluster_technique == 'meanshift'):
            bandwidth = estimate_bandwidth(self.check_depth, quantile=0.2, n_samples=500)
            self.cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(self.check_depth)

          if (self.cluster_technique == 'ransac'):
            median = np.median(self.check_depth, axis=0)
            diff = np.sum((self.check_depth - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation
            thresh = 3.5

            inliers = modified_z_score <= thresh

            self.clustered_index = np.zeros(self.masks_test[0].reshape(-1).shape).astype('int')
            self.clustered_index[self.depth_val.nonzero()] = inliers.astype('int')
            self.valid_pts_index = self.clustered_index.astype('bool')

          if (self.cluster_technique == 'kmeans' or self.cluster_technique == 'meanshift'):
            self.clustered_index = -np.ones(self.masks_test[0].reshape(-1).shape).astype('int')
            self.clustered_index[self.depth_val.nonzero()] = self.cluster.labels_
            self.center_label = np.argsort(self.cluster.cluster_centers_, axis=0)[0][0]  # get nearest label
            self.valid_pts_index = self.clustered_index == self.center_label


        # get color hists and mask info for 3D reconstruction
        if (first_masking):
          self.ID_2D += 1
          if (self.is_depth_clustered):
            color_pixs = self.color_im[self.valid_pts_index.reshape(self.masks_test[0].shape)]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.mask_data[self.valid_pts_index] = 10000 * class_score + \
                                                   100 * class_index + \
                                                   self.ID_2D

            self.make_node_data(self.ID_2D, class_name, boxes[i, :], self.color_hist, is_new=True)

          else:
            color_pixs = self.color_im[self.masks_test[i].nonzero()]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.mask_data[self.masks_test[i].reshape(-1).nonzero()] = 10000 * class_score + \
                                                                       100 * class_index + \
                                                                       self.ID_2D
            self.make_node_data(self.ID_2D, class_name, boxes[i, :], self.color_hist, is_new=True)
          comp_ID += [self.ID_2D]
        else:
          update_id = max(self.unique_ID_3D) + i + 1
          # print('update_id :{}, update_class : {}'.format(update_id, class_name))
          if (self.is_depth_clustered):
            color_pixs = self.color_im[self.valid_pts_index.reshape(self.masks_test[0].shape)]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)

            self.make_node_data(update_id, class_name, boxes[i, :], self.color_hist, is_new=True)


            self.mask_data[self.valid_pts_index] = 10000 * class_score + \
                                                   100 * class_index + \
                                                   update_id
          else:
            color_pixs = self.color_im[self.masks_test[i].nonzero()]
            self.color_hist = self.GCI.get_color_hist_kmeans(color_pixs)
            self.make_node_data(update_id, class_name, boxes[i, :], self.color_hist, is_new=True)
            self.mask_data[self.masks_test[i].reshape(-1).nonzero()] = 10000 * class_score + \
                                                                       100 * class_index + \
                                                                       update_id

          comp_ID += [update_id]

    if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
      update_masked_img, update_tsdf = 1, 1
      self.cuda_integrate_param(obs_weight, num_masks, update_masked_img, update_tsdf, first_masking, color_im, depth_im)

      self.cuda_to_cpu()
      self.draw_updated_3d_pts()

      # remove dropped idx during same node detection
      pop_list = []
      for (key, val) in self.node_data.items():
        if (not int(key) in self.unique_ID_3D):
          pop_list += [key]
      for pop_ in pop_list:
        self.node_data.pop(pop_)

      # Make relation nodes data
      if (self.node_data):
        self.make_rel_data(self.node_data.keys())
        self.draw_scene_graph(frame_num)

      self.cpu_to_cuda()


  def cuda_to_cpu(self):
    cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
    cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    self._class_vol_gpu.free()
    self._color_vol_gpu.free()

  def cpu_to_cuda(self):
    self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
    cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)
    self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
    cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

  def cuda_integrate_param(self, obs_weight, num_masks, update_masked_img, update_tsdf, fisrt_masking, color_im, depth_im):
    for gpu_loop_idx in range(self._n_gpu_loops):
      self._cuda_integrate(self._tsdf_vol_gpu,
                           self._weight_vol_gpu,
                           self._color_vol_gpu,
                           self._class_vol_gpu,
                           cuda.InOut(self.mask_data),
                           cuda.InOut(self.class_colors.reshape(-1).astype(np.float32)),
                           cuda.InOut(self._vol_dim.astype(np.float32)),
                           cuda.InOut(self._vol_origin.astype(np.float32)),
                           cuda.InOut(self.cam_intr.reshape(-1).astype(np.float32)),
                           cuda.InOut(self.cam_pose.reshape(-1).astype(np.float32)),
                           cuda.InOut(np.asarray([
                             gpu_loop_idx,
                             self._voxel_size,
                             self.im_h,
                             self.im_w,
                             self._trunc_margin,
                             obs_weight,
                             num_masks,
                             self._trunc_margin_mask,
                             update_masked_img,
                             update_tsdf,
                             fisrt_masking
                           ], np.float32)),
                           cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                           cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                           block=(self._max_gpu_threads_per_block, 1, 1),
                           grid=(
                             int(self._max_gpu_grid_dim[0]),
                             int(self._max_gpu_grid_dim[1]),
                             int(self._max_gpu_grid_dim[2]),
                           )
                           )


  def draw_updated_3d_pts(self):
    # class_vol_cpu : [detect_cnt][cls_score][cls_label][3D_instance]
    nonzero_class_vol = self._class_vol_cpu[(self._class_vol_cpu + 1).nonzero()].copy()
    ID_3D = nonzero_class_vol.astype('int') - (nonzero_class_vol / 100).astype('int') * 100
    unique_ID_3D = list(set(ID_3D))
    self.unique_ID_3D = unique_ID_3D

    if self.debug_same_node_detector:
      self.mask_centers = []
      self.cam_frustum = []
      self.class_label = []
      self.bbox_3ds = {}
      class_vol_ids = (self._class_vol_cpu).astype('int') - ((self._class_vol_cpu / 100).astype('int') * 100)
      detect_cnt = (self._class_vol_cpu / 1000000).astype('int')
      for idx in unique_ID_3D:
        ID_index = np.where(np.logical_and(class_vol_ids == idx, detect_cnt >= 1))
        # ID_index = np.where(class_vol_ids == idx)
        self.ID_index_array = np.array(ID_index)
        if (len(ID_index[0]) != 0):
          ID_3D_pose = self.ID_index_array.transpose() * self._voxel_size + self._vol_origin.reshape(-1, 3)
          ID_3D_mean = np.mean(ID_3D_pose, axis=0)
          ID_class = int(self._class_vol_cpu[ID_index][0] / 100) - int(self._class_vol_cpu[ID_index][0] / 10000) * 100
          ID_class = self.cfg.dataset.class_names[ID_class]
          self.mask_centers += [list(ID_3D_mean)]
          self.class_label += [ID_class + str(idx)]

          min_x, min_y, min_z = np.min(ID_3D_pose, axis=0)
          max_x, max_y, max_z = np.max(ID_3D_pose, axis=0)
          self.bbox_3ds[str(idx)] = [[min_x, min_y, min_z], [max_x, min_y, min_z],
                                          [max_x, max_y, min_z], [min_x, max_y, min_z],
                                          [min_x, min_y, max_z], [max_x, min_y, max_z],
                                          [max_x, max_y, max_z], [min_x, max_y, max_z]
                                          ]

          # # for debugging the individual voxel class points
          # self.ID_3D_pose = ID_3D_pose
          # self.mask_centers += ID_3D_pose.tolist()
          # self.class_label += [ID_class + str(idx)] * ID_3D_pose.shape[0]

          # make scene graph node
          self.node_data[str(idx)]['mean'] = list(ID_3D_mean)

      # find camera's 3d position in tsdf_vol
      self.T = np.linalg.inv(self.cam_pose)
      self.R = self.T[:3, :3]
      self.t = self.T[:3, 3]
      self.camera_3D_pose = np.matmul(-np.linalg.inv(self.R), self.t)
      # self.mask_centers += [self.camera_3D_pose]
      # self.class_label += ['camera']

      self.cam_frustum += [self.camera_3D_pose]
      r1 = self.R[0] * self._voxel_size
      r2 = self.R[1] * self._voxel_size
      r3 = self.R[2] * self._voxel_size
      self.r1_ = r1 / np.sqrt(sum(r1 * r1))
      self.r2_ = r2 / np.sqrt(sum(r2 * r2))
      self.r3_ = r3 / np.sqrt(sum(r3 * r3))

      a1 = -r1 -r2 + r3 + self.camera_3D_pose
      a2 = -r1 + r2 + r3 + self.camera_3D_pose
      a3 = r1 + r2 + r3 + self.camera_3D_pose
      a4 = r1 - r2 + r3 + self.camera_3D_pose
      self.cam_frustum_plane = np.array([a1, a2, a3, a4])
      self.cam_frustum += self.cam_frustum_plane.tolist()
      self.cam_connect = np.array([[0,1], [0,2], [0,3], [0,4],[1,2],[2,3],[3,4],[4,1]], dtype=np.int32)
      self.cam_centers = [self.camera_3D_pose]
      self.cam_label = ['camera']

  def make_node_data(self, obj_id, class_name, boxes, color_hist, is_new=True):
    if is_new:
      self.node_data[str(obj_id)] = {}
      self.node_data[str(obj_id)]['class'] = class_name
      self.node_data[str(obj_id)]['detection_cnt'] = 0
      x1, y1, x2, y2 = boxes
      box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
      thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + str(obj_id) +
                                    '_' + str(self.node_data[str(obj_id)]['detection_cnt']) + '.png')
      cv2.imwrite(thumbnail_path, box_im)
      self.node_data[str(obj_id)]['color_hist'] = color_hist

    else:
      self.node_data[str(obj_id)]['class'] = class_name
      self.node_data[str(obj_id)]['detection_cnt'] += 1
      x1, y1, x2, y2 = boxes
      box_im = cv2.cvtColor(self.color_im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
      thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + str(obj_id) +
                                    '_' + str(self.node_data[str(obj_id)]['detection_cnt']) + '.png')
      cv2.imwrite(thumbnail_path, box_im)
      self.node_data[str(obj_id)]['color_hist'] = color_hist

  def make_rel_data(self, comp_ID):
    cam_pose = self.camera_3D_pose
    self.rel_data = {}
    for i, sub_idx in enumerate(comp_ID):
      for j, obj_idx in enumerate(comp_ID):
        rel = []
        th = self._voxel_size * 3
        if (sub_idx > obj_idx):
          if (str(sub_idx) in self.node_data.keys() and str(obj_idx) in self.node_data.keys()):
            sub = self.node_data[str(sub_idx)]
            obj = self.node_data[str(obj_idx)]
            sub_pose = sub['mean']
            obj_pose = obj['mean']
            sub_cam = sub_pose - cam_pose
            obj_cam = obj_pose - cam_pose
            s1 = np.dot(sub_cam, self.r1_)
            s2 = np.dot(sub_cam, self.r2_)
            s3 = np.dot(sub_cam, self.r3_)

            o1 = np.dot(obj_cam, self.r1_)
            o2 = np.dot(obj_cam, self.r2_)
            o3 = np.dot(obj_cam, self.r3_)

            if (s1 - o1 > th):
              rel += ['right']
            elif (s1 - o1 < -th):
              rel += ['left']
            if (s2 - o2 > th):
              rel += ['up']
            elif (s2 - o2 < -th):
              rel += ['down']
            if (s3 - o3 > th):
              rel += ['behind']
            elif (s3 - o3 < -th):
              rel += ['forward']

            self.rel_data[sub['class'] + '_' + str(sub_idx) + '/' +
                          obj['class'] + '_' + str(obj_idx)] = rel

  def draw_scene_graph(self, frame_num):
    # Draw scene graph
    sg = Digraph('structs', format='pdf')
    detect_th = 0
    tomato_rgb = [236, 93, 87]
    blue_rgb = [81, 167, 250]
    pale_rgb = [112, 191, 64]
    tomato_hex = webcolors.rgb_to_hex(tomato_rgb)
    blue_hex = webcolors.rgb_to_hex(blue_rgb)
    pale_hex = webcolors.rgb_to_hex(pale_rgb)

    for node_idx in self.node_data.keys():
      idx = str(node_idx)
      node = self.node_data[idx]
      obj_cls = node['class']
      detect_num = node['detection_cnt']
      if (detect_num >= detect_th):
        sg.node(obj_cls + '_' + idx, shape='box', style='filled, rounded', label=obj_cls + '_' + idx,
                margin='0.11, 0.0001', width='0.11', height='0', fillcolor=tomato_hex,
                fontcolor='black')
        sg.node('attribute_pose_' + idx, shape='box', style='filled, rounded',
                label='(' + str(round(node['mean'][0], 2)) + ',' +
                      str(round(node['mean'][1], 2)) + ',' +
                      str(round(node['mean'][2], 2)) + ')',
                margin='0.11,0.0001', width='0.11', height='0',
                fillcolor=blue_hex, fontcolor='black'
                )
        sg.node('attribute_color_' + idx, shape='box', style='filled, rounded',
                label=str(node['color_hist'][0][1]),
                margin='0.11, 0.0001', width='0.11', height='0',
                fillcolor=node['color_hist'][0][0], fontcolor='black')
        thumbnail_path = os.path.join(self.bbox_path, 'thumbnail_' + str(idx) +
                                    '_' + str(int(self.node_data[str(idx)]['detection_cnt']/2)) + '.png')
        sg.node('thumbnail_'+idx, shape='box', label='.', image=thumbnail_path)
        # For one shot image
        # sg.node('thumbnail_' + idx, shape='box', label='.',
        #         image=node['thumbnail'][0])
        sg.edge(obj_cls + '_' + idx, 'attribute_pose_' + idx)
        sg.edge(obj_cls + '_' + idx, 'attribute_color_' + idx)
        sg.edge(obj_cls + '_' + idx, 'thumbnail_' + idx)

    for i, (key, value) in enumerate(self.rel_data.items()):
      sub = key.split('/')[0]
      obj = key.split('/')[1]
      sub_id = sub.split('_')[1]
      obj_id = obj.split('_')[1]
      # Draw scene graph relation if objects detected more than detect_th
      if ((self.node_data[str(sub_id)]['detection_cnt'] >= detect_th) and (
              self.node_data[str(obj_id)]['detection_cnt'] >= detect_th)):
        if (value):
          # check value has some info
          rel = ''
          for v in range(len(value)):
            if (v != len(value) - 1):
              rel = rel + value[v] + ','
            else:
              rel += value[v]
          sg.node('rel' + str(i), shape='box', style='filled, rounded', fillcolor=pale_hex,
                  fontcolor='black', margin='0.11, 0.0001', width='0.11', height='0',
                  label=rel)
          sg.edge(sub, 'rel' + str(i))
          sg.edge('rel' + str(i), obj)
    # sg.render(os.path.join(save_path, 'scene_graph'+str(f_idx)), view=True)
    sg.format = 'png'
    # sg.size = "480,640"
    sg.render(os.path.join(self.scene_graph_path, 'scene_graph' + str(frame_num)), view=False)
    self.f_idx += 1


  def get_volume(self):
    if self.gpu_mode:
      cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
      cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    return self._tsdf_vol_cpu, self._color_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol = self.get_volume()


    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def get_mesh(self):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))
