import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp


class ScannetVis(QWidget):
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, scan, rgb_names, depth_names, pose_names, offset=0, skip_im=10, mesh_plot=True, parent=None):
        super(ScannetVis, self).__init__(parent=parent)

        self.scan = scan
        self.rgb_names = rgb_names
        self.depth_names = depth_names
        self.pose_names = pose_names
        self.offset = offset
        self.offset_prev = offset
        self.skip_im = skip_im
        self.mesh_plot = mesh_plot

        self.keyboard_inputs = None
        self.total = len(self.rgb_names)

        self.checkBox_list = []
        self.checkBox_with_3D = []

        self.reset()
        self.initUI()
        self.update_scan()


    def initUI(self):
        self.setStyleSheet("background-color: white;")
        self.principalLayout = QHBoxLayout(self)

        ''' left left Frame : RGB with yolact & depth frame '''
        self.left2Frame = QFrame(self)
        self.left2Frame.setFrameShape(QFrame.StyledPanel)
        self.left2Frame.setFrameShadow(QFrame.Raised)
        self.vertical2Layout = QVBoxLayout(self.left2Frame)
        # self.vertical2Layout.setSpacing(0)
        self.principalLayout.addWidget(self.left2Frame)

        # self.vertical2_1Layout = QVBoxLayout(self.left2Frame)
        # self.vertical2Layout.addWidget(self.left2Frame)
        # add rgb depth
        self.img_canvas.create_native()
        self.img_canvas.native.setMinimumSize(320, 480)
        self.vertical2Layout.addWidget(self.img_canvas.native)

        ''' left Frame : 3D reconstructed Scene '''
        self.leftFrame = QFrame(self)
        self.leftFrame.setFrameShape(QFrame.StyledPanel)
        self.leftFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.leftFrame)
        # self.verticalLayout.setSpacing(0)
        self.principalLayout.addWidget(self.leftFrame)

        self.canvas.create_native()
        self.canvas.native.setMinimumSize(640, 480)
        self.verticalLayout.addWidget(self.canvas.native)

        ''' left center Frame : 3D Scene graph'''
        self.SGFrame = QFrame(self)
        self.SGFrame.setFrameShape(QFrame.StyledPanel)
        self.SGFrame.setFrameShadow(QFrame.Raised)
        # self.verticalSGLayout = QVBoxLayout(self.SGFrame)
        # self.verticalLayout.setSpacing(0)
        self.principalLayout.addWidget(self.SGFrame)

        self.scene_graph_canvas.create_native()
        self.scene_graph_canvas.native.setMinimumSize(640, 480)
        self.verticalLayout.addWidget(self.scene_graph_canvas.native)

        ''' center Frame : control pannel '''
        self.keyFrame = QFrame(self)
        self.keyFrame.setFrameShape(QFrame.StyledPanel)
        self.keyFrame.setFrameShadow(QFrame.Raised)
        self.keysverticalLayout = QVBoxLayout(self.keyFrame)

        self.label1 = QLabel("To navigate: "
                             "\n   n: next (next scan) "
                             "\n   s: start (start processing sequential rgb-d images)"
                             "\n   p: pause (pause processing)"
                             "\n   q: quit (exit program)"

                             "\n\n To control 3D view: "
                             "\n   LMB: orbits the view around its center point"
                             "\n   RMB or scroll: change scale_factor (i.e. zoom level)"
                             "\n   SHIFT + LMB: translate the center point"
                             "\n   SHIFT + RMB: change FOV")
        self.label2 = QLabel("To find specific objects in 3D Space : ")
        # self.keysverticalLayout.addWidget(self.label1)
        # self.keysverticalLayout.addWidget(self.label2)
        self.vertical2Layout.addWidget(self.label1)
        self.vertical2Layout.addWidget(self.label2)

        self.le = QLineEdit(self)
        self.vertical2Layout.addWidget(self.le)

        self.spb = QPushButton('search', self)
        self.vertical2Layout.addWidget(self.spb)
        self.spb.clicked.connect(self.search_button_click)

        self.cpb = QPushButton('clear', self)
        self.vertical2Layout.addWidget(self.cpb)
        self.cpb.clicked.connect(self.clear_button_click)

        self.verticalLayoutR = QVBoxLayout()
        self.verticalLayoutR.addWidget(self.keyFrame)
        self.verticalLayoutR.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutR.setSpacing(0)
        self.principalLayout.addLayout(self.verticalLayoutR)

        ''' Right Frame : result images of searched objects '''
        self.rightFrame = QFrame(self)
        self.rightFrame.setFrameShape(QFrame.StyledPanel)
        self.rightFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayoutRight = QVBoxLayout(self.rightFrame)
        self.verticalLayoutRight.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutRight.setSpacing(0)
        self.principalLayout.addWidget(self.rightFrame)

        self.setLayout(self.principalLayout)
        self.setWindowTitle('Searching objects')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        ''' 3D points cloud or mesh SceneCanvas '''
        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # add point cloud views
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)

        # Camera location settings
        self.scene_cam = vispy.scene.cameras.BaseCamera()
        # self.scene_cam.center = (-10, -10, 10)
        # self.scan_view.add(self.scene_cam)
        # self.scene_cam.pre_transform.set_range()

        canvas2 = vispy.app.Canvas()
        w = QMainWindow()
        widget = QWidget()
        w.setCentralWidget(widget)
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(canvas2.native)
        widget.layout().addWidget(QPushButton())
        w.show()

        self.scan_vis = visuals.Mesh()
        self.scan_vis_mean = visuals.Line()
        self.scan_vis_cam = visuals.Line()
        self.scan_bbox_3d = visuals.Line()
        self.label_vis = visuals.Text()

        self.scan_view.add(self.scan_vis)
        self.scan_view.add(self.scan_vis_mean)
        self.scan_view.add(self.scan_vis_cam)
        self.scan_view.add(self.scan_bbox_3d)
        self.scan_view.add(self.label_vis)

        self.scan_view.camera = 'arcball'
        self.tr = self.scan_vis.transforms.get_transform(map_from='visual', map_to='canvas')
        # self.scan_view.camera = self.scene_cam
        # self.scan_view.camera = 'arcball' , 'turntable'
        # self.scan_view.camera.transform.rotate(90, (0,1,0))

        visuals.XYZAxis(parent=self.scan_view.scene)

        ''' 2D images SceneCanvas '''
        # img canvas size
        self.canvas_W = 320
        self.canvas_H = 280
        self.multiplier = 2

        ''' new canvas for RGB & Depth img '''
        self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                      size=(self.canvas_W, self.canvas_H * self.multiplier))
        self.img_grid = self.img_canvas.central_widget.add_grid()
        # interface (n next, s start, p pause, q quit, )
        self.img_canvas.events.key_press.connect(self.key_press)
        self.img_canvas.events.draw.connect(self.draw)

        # add rgb views
        self.rgb_img_raw_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.rgb_img_raw_view, 0, 0)
        self.rgb_img_raw_vis = visuals.Image(cmap='viridis')
        self.rgb_img_raw_view.add(self.rgb_img_raw_vis)

        # add a view for the depth
        self.depth_img_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.depth_img_view, 1, 0)
        self.depth_img_vis = visuals.Image(cmap='viridis')
        self.depth_img_view.add(self.depth_img_vis)

        ''' new canvas for 3D scene graph img '''
        self.scene_graph_canvas = SceneCanvas(keys='interactive', show=True,
                                      size=(640, 480))
        self.scene_graph_grid = self.scene_graph_canvas.central_widget.add_grid()
        self.scene_graph_canvas.events.key_press.connect(self.key_press)
        self.scene_graph_canvas.events.draw.connect(self.draw)

        # add a view for 3D scene graphs
        self.scene_graph_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.scene_graph_canvas.scene)
        self.scene_graph_grid.add_widget(self.scene_graph_view, 0, 0)
        self.scene_graph_vis = visuals.Image(cmap='viridis')
        self.scene_graph_view.add(self.scene_graph_vis)


    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0


    def update_yolact(self):
        title = "scan " + str(self.offset)

        # draw color & depth image
        self.img_canvas.title = title

        _, _, _, _ = self.scan.open_scan(self.rgb_names[self.offset],
                                         self.depth_names[self.offset],
                                         self.pose_names[self.offset],
                                         self.offset,
                                         recon=False)

        text_str = 'Frame %d ' % (self.offset)
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        masked_img = self.scan.masked_img.copy()
        masked_img = cv2.resize(masked_img, (320, 240), interpolation=cv2.INTER_AREA)

        x1, y1 = 0, 0
        text_pt = (x1, y1 + 15)
        text_color = [255, 255, 255]
        color = [0, 0, 0]
        cv2.rectangle(masked_img, (x1, y1), (x1 + text_w, y1 + text_h + 4), color, -1)
        cv2.putText(masked_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        self.rgb_img_raw_vis.set_data(masked_img)
        self.rgb_img_raw_vis.update()

        depth_img = cv2.resize(self.scan.depth_im.copy(), (320, 240), interpolation=cv2.INTER_AREA)
        self.depth_img_vis.set_data(depth_img)
        self.depth_img_vis.update()

    def update_3d_recon(self):
        title = "scan " + str(self.offset)
        if (self.offset % self.skip_im == 0):
            start_time = time.time()
            verts, faces, norms, colors = self.scan.open_scan(self.rgb_names[self.offset],
                                                              self.depth_names[self.offset],
                                                              self.pose_names[self.offset],
                                                              self.offset,
                                                              recon=True
                                                              )
            self.verts, self.faces, self.norms, self.colors = verts, faces, norms, colors

            self.canvas.title = title
            self.scan_vis.set_data(vertices=verts,
                                   faces=faces,
                                   vertex_colors=colors/255.)
            self.scan_vis.update()

            #if self.scan.num_dets_to_consider > 0 and not self.scan.use_gpu:
            if self.scan.num_dets_to_consider > 0 and self.scan.tsdf_vol.debug_same_node_detector:
                self.mean_pose = np.array(self.scan.tsdf_vol.mask_centers)
                self.scan_vis_mean.set_data(
                    self.mean_pose,
                    color='red',
                    width=3,
                    connect='strip')
                self.scan_vis_mean.update()

                # find object's position and visualize
                self.label_vis.text = self.scan.tsdf_vol.class_label
                self.label_vis.pos = self.mean_pose
                self.label_vis.font_size = int(40)

            self.cam_frustum = np.array(self.scan.tsdf_vol.cam_frustum)
            self.scan_vis_cam.set_data(
                self.cam_frustum,
                color='blue',
                width=3,
                connect=self.scan.tsdf_vol.cam_connect
            )
            self.scan_vis_cam.update()
            if ('camera' in self.label_vis.text):
                self.label_vis.text.pop()
                self.label_vis.pos = self.label_vis.pos[:-1, :]
            self.label_vis.text += self.scan.tsdf_vol.cam_label
            self.label_vis.pos = np.append(self.label_vis.pos, self.scan.tsdf_vol.cam_centers, axis=0)

            # Draw Scene graph images
            generated_scene_graph_file = os.path.join(self.scan.tsdf_vol.scene_graph_path,
                                                      'scene_graph' + str(self.offset)+'.png')
            if os.path.exists(generated_scene_graph_file):
                print('Draw scene graph{}'.format(self.offset))
                sg_img = cv2.cvtColor(cv2.imread(generated_scene_graph_file),
                                      cv2.COLOR_BGR2RGB)
                self.sg_img = cv2.resize(sg_img, (640, 480), interpolation=cv2.INTER_AREA)
                self.scene_graph_vis.set_data(self.sg_img)
                self.scene_graph_vis.update()

            print("--- %s seconds of %d to %d images---" % (time.time() - start_time, self.offset-self.skip_im+1, self.offset))
            print("--- fps : {} ---".format(self.skip_im / (time.time() - start_time)))

    def update_scan(self):
        # update_yolact images
        self.update_yolact()

        # Reconstruct 3D Scene and detect same nodes or not
        self.update_3d_recon()

    def update_seq_scan(self):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
            self.img_canvas.events.key_press.unblock()
        if self.scene_graph_canvas.events.key_press.blocked():
            self.scene_graph_canvas.events.key_press.unblock()
        if(self.start):
            self.offset += 1
            self.update_yolact()
            self.update_3d_recon()

            self.canvas.scene.update()
            self.img_canvas.scene.update()
            self.scene_graph_canvas.update()
            self.canvas.on_draw(None)
            self.img_canvas.on_draw(None)
            self.scene_graph_canvas.on_draw(None)

    # interface
    def key_press(self, event):
        self.keyboard_inputs = event.key
        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0
            self.update_scan()

        elif event.key == 'S':
            # Start to process RGB-D sequences
            self.start = True
            self.timer1 = vispy.app.Timer(0.033, connect=self.on_timer1, start=True)
            self.timer2 = vispy.app.Timer(0.033, connect=self.on_timer2, start=True)

        elif event.key == 'P':
            # Pause to process RGB sequences
            self.start = False

        elif event.key == 'U':
            # test when updated draw function
            self.canvas.scene.update()
            self.img_canvas.scene.update()
            self.scene_graph_canvas.update()

        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def on_timer1(self, event):
        # self.update_seq_scan()
        if(self.start):
            self.offset += 1
            self.update_yolact()

    def on_timer2(self, event):
        if (self.start):
            # self.offset += 1
            self.update_3d_recon()

    def search_button_click(self):
        print('searching object : {}'.format(self.le.text()))
        objects_dict = self.scan.tsdf_vol.node_data

        is_obj_exist = []

        self.clear_searched_items(self.verticalLayoutRight)

        for key, val in objects_dict.items():
            if (val['class'] == self.le.text()):
                print('find {}'.format(self.le.text()))

                thumbnail_path = os.path.join(self.scan.tsdf_vol.bbox_path, 'thumbnail_' + str(key) +
                                              '_' + str(int(objects_dict[str(key)]['detection_cnt'] / 2)) + '.png')
                cv2_img = cv2.cvtColor(cv2.imread(thumbnail_path), cv2.COLOR_BGR2RGB)
                image = QImage(cv2_img.data, cv2_img.shape[1], cv2_img.shape[0], cv2_img.strides[0], QImage.Format_RGB888)
                image_frame = QLabel()
                image_frame.setPixmap(QPixmap.fromImage(image))
                self.verticalLayoutRight.addWidget(image_frame)

                checkBox = QCheckBox(val['class'] + str(key))
                self.checkBox_list += [[checkBox, val['class'], str(key)]]

                scan_bbox_3d = visuals.Line()
                self.checkBox_with_3D += [scan_bbox_3d]
                self.scan_view.add(scan_bbox_3d)

                checkBox.stateChanged.connect(self.checkBoxState)

                # searched_obj = QLabel(val['class'] + str(key))
                self.verticalLayoutRight.addWidget(checkBox)
                is_obj_exist += [True]

        if(not is_obj_exist):
            searched_obj = QLabel("Nothing was found!")
            self.verticalLayoutRight.addWidget(searched_obj)
        else:
            searched_obj = QLabel("Check box if you want to find objects in 3D Scene.")
            self.verticalLayoutRight.addWidget(searched_obj)

    def clear_button_click(self):
        print('clear previous searched object')
        self.clear_searched_items(self.verticalLayoutRight)

    def clear_searched_items(self, layout):
        # reset rearching results widget
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                continue

            w = item.widget()
            if w:
                w.deleteLater()

        # reset visuals.Line for 3D BBox of searched objects
        for i, check in enumerate(self.checkBox_list):
            self.checkBox_with_3D[i].parent = None
            self.checkBox_with_3D[i] = visuals.Line()
            self.scan_view.add(self.checkBox_with_3D[i])

        self.checkBox_list = []
        self.checkBox_with_3D = []

    def checkBoxState(self):
        # checkBox_list is composed of [QcheckBox, class_name, class_3D_ID]
        for i, check in enumerate(self.checkBox_list):
            if check[0].isChecked() == True:
                print('checked!!!')
                # Find 3D BBox in 3D Scene Canvas\
                bbox_3d = np.array(self.scan.tsdf_vol.bbox_3ds[check[2]])
                bbox_connect = np.array([[0,1], [1,2], [2,3], [3,0],
                                         [4,5], [5,6], [6,7], [7,4],
                                         [0,4], [1,5], [2,6], [3,7]])
                self.checkBox_with_3D[i].set_data(bbox_3d,
                    color='green',
                    width=3,
                    connect=bbox_connect)
            else:
                self.checkBox_with_3D[i].parent = None
                self.checkBox_with_3D[i] = visuals.Line()
                self.scan_view.add(self.checkBox_with_3D[i])

    def draw(self, event):
        # print('draw states!!')
        # print('event key: {}'.format(self.keyboard_inputs))
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
            self.img_canvas.events.key_press.unblock()
        if self.scene_graph_canvas.events.key_press.blocked():
            self.scene_graph_canvas.events.key_press.unblock()

        if self.keyboard_inputs == 'P':
            # Pause to process RGB sequences
            self.start = False
        # if self.keyboard_inputs == 'S':
        #     self.update_seq_scan()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.img_canvas.close()
        self.scene_graph_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.use_app(backend_name="PyQt5", call_reuse=True)
        vispy.app.run()