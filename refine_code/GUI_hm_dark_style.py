import cv2
import sys
import os.path as osp

import time
from matplotlib import pyplot as plt
import numpy as np
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;
import os
from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.kmedoids import kmedoids;
import pickle
from pyclustering.utils import read_sample;
from pyclustering.utils import timedcall;
import functools
import transfer_lab
import copy
import math
import qdarkstyle

from PyQt5.QtWidgets import QApplication,QScrollArea,QColorDialog, QWidget,QMainWindow,QLabel, QPushButton,QFormLayout,QMessageBox,QLineEdit,QAction, QFileDialog,QVBoxLayout,QHBoxLayout,QGroupBox,QFrame
from PyQt5.QtGui import QIcon,QImage,QPixmap,QPalette,QColor
from PyQt5.QtCore import pyqtSlot,Qt, QThread,pyqtSignal

class ColorChangeHue(QMainWindow):

    def __init__(self, parent_win, dom_colors):
        super().__init__()
        self.title = 'Interactively change the color by hue matching'
        self.left = 50
        self.top = 50
        self.width = 480
        self.height = 150
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)
        self.mainlayout = QVBoxLayout()
        self.wid.setLayout(self.mainlayout)
        print(dom_colors)
        self.h_dom_colors = []
        for dc in dom_colors:
            self.h_dom_colors.append(transfer_lab.rgb2hue_opencv(transfer_lab.lab2rgb_opencv(dc))[0]*2)
        self.parent_win = parent_win
        self.add_hue_matchings()
        self.source_hue_idx=None
        self.target_hue_idx=None
        self.src_tar_mp = {}

        self.setStyleSheet("QLabel{background:white;}"
                           "QLabel{color:rgb(100,100,100,250);font-size:15px;font-weight:bold;font-family:Roman times;}"
                           "QLabel:hover{color:rgb(100,100,100,120);}")

    def add_hue_matchings(self):
        self.top_hue_layout = QHBoxLayout()
        self.mid_hue_layout = QHBoxLayout()
        self.bottom_hue_layout = QHBoxLayout()
        self.mainlayout.addLayout(self.top_hue_layout)
        self.mainlayout.addLayout(self.mid_hue_layout)
        self.mainlayout.addLayout(self.bottom_hue_layout)
        bin_size=10
        self.binsz = bin_size
        for idx, w in enumerate(range(0,360,bin_size)):
            mid_h = int((idx + 0.5) * bin_size)
            if mid_h >= 360:
                break
            q = QLabel()
            q.setAutoFillBackground(True)
            q.setText("  ")

            q.setMargin(5)
            p = q.palette()
            ss = False
            for hu_range in range(int(mid_h-bin_size*0.5), int(mid_h+bin_size*0.5)):
                if hu_range in self.h_dom_colors:
                    ss=True
            if ss:
                # p.setColor(q.backgroundRole(),QColor.fromHsl(mid_h,255,128))
                q.setStyleSheet(
                    "background-color: %s" % (QColor.fromHsl(mid_h,255,128).name()))
            else:
                # p.setColor(q.backgroundRole(), QColor.fromHsl(mid_h, 255, 128,alpha=0))
                q.setStyleSheet(
                    "background-color: %s" % (QColor.fromHsl(mid_h, 255, 128,alpha=0).name(QColor.HexArgb)))
            q.setPalette(p)
            q.mousePressEvent = functools.partial(self.choose_source_hue, source_object=q, index=idx)
            self.top_hue_layout.addWidget(q)

        for idx, w in enumerate(range(0,360,bin_size)):
            if int((idx + 0.5) * bin_size) >= 360:
                break
            q = QLabel()
            q.setAutoFillBackground(True)
            q.setText("  ")

            q.setMargin(5)
            p = q.palette()
            # p.setColor(q.backgroundRole(),QColor.fromHsl(int((idx+0.5)*bin_size), 255, 128, alpha=0))
            q.setPalette(p)
            qc = QColor.fromHsl(int((idx + 0.5) * bin_size), 255, 128, alpha=0)
            q.setStyleSheet("background-color: %s" % (qc.name(QColor.HexArgb)))
            self.mid_hue_layout.addWidget(q)

        for idx, w in enumerate(range(0,360,bin_size)):
            if int((idx + 0.5) * bin_size) >= 360:
                break
            q = QLabel()
            q.setAutoFillBackground(True)

            q.setMargin(5)
            p = q.palette()
            p.setColor(q.backgroundRole(),QColor.fromHsl(int((idx+0.5)*bin_size),255,128))
            q.setPalette(p)
            q.setStyleSheet("background-color: %s" % (QColor.fromHsl(int((idx + 0.5) * bin_size), 255, 128).name()))
            q.mousePressEvent = functools.partial(self.choose_target_hue, source_object=q, index=idx)
            self.bottom_hue_layout.addWidget(q)

    def choose_source_hue(self,event,source_object=None,index=None):
        self.source_hue_idx = index
        pass
    def choose_target_hue(self, event,source_object=None,index=None):
        self.target_hue_idx = index
        if self.source_hue_idx is not None:
            q = self.mid_hue_layout.itemAt(self.source_hue_idx).widget()
            p = q.palette()
            p.setColor(q.backgroundRole(), QColor.fromHsl(int((self.target_hue_idx + 0.5) * self.binsz), 255, 128, alpha=255))
            q.setPalette(p)
            qc = QColor.fromHsl(int((self.target_hue_idx + 0.5) * self.binsz), 255, 128, alpha=255)
            q.setStyleSheet("background-color: %s" % (qc.name(QColor.HexArgb)))
            self.src_tar_mp[self.source_hue_idx] = self.target_hue_idx

            #更新目标颜色 并刷新结果
            self.parent_win.update_target_colors(self.src_tar_mp,self.binsz)
        pass
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Translucent Image Recoloring through Homography Estimation'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
        self._dom_colors = None

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)
        self.mainLayout = QHBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.formlayout = QFormLayout()
        self.rightLayout.addLayout(self.formlayout)

        self.clayout = QHBoxLayout()
        self.dom_color_layout = QVBoxLayout()
        self.tar_color_layout = QVBoxLayout()
        self.clayout.addLayout(self.dom_color_layout)
        self.clayout.addLayout(self.tar_color_layout)
        self.rightLayout.addLayout(self.clayout)

        self.rightLayout.addStretch(1)

        self.scroll = QScrollArea()
        self.scroll.setLayout(self.rightLayout)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(600)
        self.scroll.setFrameStyle(QFrame.NoFrame)
        self.rr_new = QVBoxLayout()
        self.rr_new.addWidget(self.scroll)


        self.wid.setLayout(self.mainLayout)

        self.imageView = QLabel("waiting to read image")

        qb = QGroupBox("Image view")
        qb.setLayout(self.leftLayout)
        self.leftLayout.addWidget(self.imageView)
        qb2 = QGroupBox("Color")
        qb2.setLayout(self.rr_new)
        self.mainLayout.addWidget(qb, 2)
        self.mainLayout.addWidget(qb2, 1)
        # self.mainLayout.addLayout(self.rightLayout,1)
        # self.mainLayout.addLayout(self.rr_new, 1)
        # button = QPushButton("button")
        # button.setToolTip("this is an example.")
        # button.move(100, 70)
        # button.clicked.connect(self.on_click)
        # self.leftLayout.addWidget(button)

        # self.textbox = QLineEdit()
        # self.textbox.move(20,20)
        # self.textbox.resize(280,40)
        # self.textbox.setText("8")
        # self.formlayout.addRow(QLabel("Cluster Number:"),self.textbox)

        mainmenu = self.menuBar()
        filemenu = mainmenu.addMenu("File")
        imagemenu = mainmenu.addMenu("Image")
        helpmenu = mainmenu.addMenu("Help")

        openImageButton = QAction('Open', self)
        openImageButton.setShortcut('Ctrl+O')
        openImageButton.setToolTip("Open Image")
        openImageButton.triggered.connect(self.open_image)
        filemenu.addAction(openImageButton)

        exitButton = QAction('Exit',self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip("Exit app")
        exitButton.triggered.connect(self.close)
        filemenu.addAction(exitButton)
        self.add_act_short(imagemenu, "Find Dom Colors","Ctrl+F","FDC",self.find_dom_colors)
        self.add_act_short(imagemenu, "Change Color by Hue","Ctrl+H","CCH",self.change_color_by_hue)
        self.add_act_short(imagemenu, "Save result","Ctrl+S","Save the debuged_result",self.save_image_domc_tarc)
        self.add_act_short(helpmenu, "About","","About",self.about_software)
        self.show()

    def add_act_short(self,parent_menu,name,shortcut,tooltip,triggered_func):
        tempAction = QAction(name,self)
        tempAction.setShortcut(shortcut)
        tempAction.setToolTip(tooltip)
        tempAction.triggered.connect(triggered_func)
        parent_menu.addAction(tempAction)

    def clear_layout(self,lay_out):
        for i in reversed(range(lay_out.count())):
            lay_out.itemAt(i).widget().setParent(None)

    def layout_widgets(self,layout):
        return (layout.itemAt(i) for i in range(layout.count()))

    @pyqtSlot()
    def save_image_domc_tarc(self):
        # save resulted Image 带时间戳
        # save domc image File & pickle File
        # save tarc image File & pickle File
        base_dir = osp.abspath(__file__)
        tre_dir = osp.join(osp.dirname(base_dir), "debug_result")
        file_name_w_ext = osp.basename(self.fileName)
        file_name, file_ext = osp.splitext(file_name_w_ext)
        if not osp.exists(osp.join(tre_dir,file_name)):
            os.mkdir(osp.join(tre_dir,file_name))
        re_dir = osp.join(tre_dir,file_name)
        time_str = str(int(time.time()))
        source_image_name = file_name+time_str+"source"+time_str+file_ext

        result_image_name = file_name+time_str+"result_"+time_str+file_ext
        dom_pickle_name = file_name+time_str+"dom_pickle_"+time_str
        domc_file_name = file_name+time_str+"dom_color_"+time_str+file_ext
        tar_pickle_name = file_name+time_str+"tar_pickle_"+time_str
        tarc_file_name = file_name+time_str+"tar_color_"+time_str+file_ext
        print(osp.join(re_dir,result_image_name))
        cv2.imwrite(osp.join(re_dir,result_image_name), cv2.cvtColor(self.reimage_rgb,cv2.COLOR_RGB2BGR))
        pickle.dump(self._dom_colors,open(osp.join(re_dir,dom_pickle_name),"wb"))
        pickle.dump(self._tar_colors,open(osp.join(re_dir,tar_pickle_name),"wb"))



        dom_ccs = self._dom_colors[:-1]
        tar_ccs = self._tar_colors[:-1]
        image_width = 500
        image_w_sep = 0.03
        image_h_t_b = 0.03
        lc_w = (1 - image_w_sep * (len(dom_ccs) + 1)) / len(dom_ccs) * image_width
        image_height = int(lc_w*(1+image_h_t_b*4))

        dom_ccs_img = np.zeros((image_height,image_width,3))
        tar_ccs_img = np.zeros((image_height,image_width,3))
        dom_ccs_img[:,:,:] = 255
        tar_ccs_img[:,:,:] = 255
        for ix,(dc, tc) in enumerate(zip(dom_ccs,tar_ccs)):
            dc_rgb = transfer_lab.lab2rgb_opencv(dc)
            tc_rgb = transfer_lab.lab2rgb_opencv(tc)

            top_h = int(image_h_t_b*image_height)
            bottom_h = int(image_height - image_h_t_b*image_height*4)
            c_w = (1-image_w_sep*(len(dom_ccs)+1))/len(dom_ccs)*image_width
            left_w = int((ix+1)*(image_width*image_w_sep) + ix*c_w)
            right_w = int(left_w+c_w)
            dom_ccs_img[top_h*3:bottom_h,left_w:right_w] = dc_rgb
            tar_ccs_img[top_h*3:bottom_h,left_w:right_w] = tc_rgb
            print(dc != tc)
            print(dc,tc)
            if any(dc != tc):
                tar_ccs_img[-top_h*3:-top_h, left_w:right_w] = tc_rgb

            cv2.imwrite(osp.join(re_dir,domc_file_name),cv2.cvtColor(dom_ccs_img.astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite(osp.join(re_dir,tarc_file_name),cv2.cvtColor(tar_ccs_img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        pass

    @pyqtSlot()
    def about_software(self):
        pass

    @pyqtSlot()
    def change_color_by_hue(self):
        print("change colro by hue")
        self.change_color_by_hue_win.show()

    @pyqtSlot()
    def find_dom_colors(self):
        print(self.dom_color_layout.count())

        self.fdm_thread = FindDomColorsThread(self.cvImage)
        self.fdm_thread.dom_colors.connect(self.update_dom_colors)
        self.fdm_thread.start()

    def update_dom_colors(self,dom_colors_):
        print("update dom_colors",dom_colors_)
        self._dom_colors = dom_colors_
        self._tar_colors = copy.deepcopy(dom_colors_)
        self.change_color_by_hue_win = ColorChangeHue(self,dom_colors_)
        self.clear_layout(self.dom_color_layout)
        self.clear_layout(self.tar_color_layout)

        c_num = len(dom_colors_)
        for i in range(c_num):
            self.dom_color_layout.addWidget(QLabel("dom_c%s" % (i)))
            self.tar_color_layout.addWidget(QLabel("tar_c%s" % (i)))

        for idx, w in enumerate(self.layout_widgets(self.dom_color_layout)):
            w.widget().setAutoFillBackground(True)
            w.widget().setText("  ")

            w.widget().setMargin(15)
            p = w.widget().palette()
            r,g,b = transfer_lab.lab2rgb_opencv(dom_colors_[idx])
            p.setColor(w.widget().backgroundRole(),QColor.fromRgb(r,g,b))
            w.widget().setPalette(p)
            w.widget().setStyleSheet("background-color: %s" % (QColor.fromRgb(r, g, b).name()))

        for idx, w in enumerate(self.layout_widgets(self.tar_color_layout)):
            w.widget().setAutoFillBackground(True)
            w.widget().setText("")

            w.widget().setMargin(15)
            p = w.widget().palette()
            r,g,b = transfer_lab.lab2rgb_opencv(dom_colors_[idx])
            p.setColor(w.widget().backgroundRole(),QColor.fromRgb(r,g,b))
            w.widget().setPalette(p)
            w.widget().setStyleSheet("background-color: %s" % (QColor.fromRgb(r,g,b).name()))
            w.widget().mousePressEvent = functools.partial(self.choose_color, source_object=w.widget(),index=idx)

    def get_aff_23(self, color_array):
        srct_ab = []
        for sc in color_array:
            srct_ab.append([sc[1], sc[2]])
        return srct_ab
    def update_target_colors(self,src_tar_map,binsz):
        # 根据 hue范围进行变色
        for k, v in src_tar_map.items():
            for idx,tc in enumerate(self._dom_colors):
                hsl = transfer_lab.rgb2hue_opencv(transfer_lab.lab2rgb_opencv(tc))
                h360 = hsl[0]*2
                if h360 >= k*binsz and h360 <= (k+1)*binsz:
                    tar_h360 = v*binsz
                    delta = tar_h360-k*binsz
                    h360 += delta
                    hsl[0] = int(h360/2)
                    self._tar_colors[idx] = transfer_lab.rgb2lab_opencv(transfer_lab.hue2rgb_opencv(hsl))
        for idx, w in enumerate(self.layout_widgets(self.tar_color_layout)):

            p = w.widget().palette()
            r,g,b = transfer_lab.lab2rgb_opencv(self._tar_colors[idx])
            p.setColor(w.widget().backgroundRole(),QColor.fromRgb(r,g,b))
            w.widget().setPalette(p)
            w.widget().setStyleSheet("background-color: %s" % (QColor.fromRgb(r, g, b).name()))
        self.update_image()

        pass

    def update_image(self):
        """根据当前的颜色变化更新图片，如果无变化则不更新"""
        print(self._tar_colors)
        if any(self._dom_colors[-1] != np.array(transfer_lab.rgb2lab_opencv([255,255,255]))):
            self._dom_colors.append(transfer_lab.rgb2lab_opencv([255,255,255]))

        if any(self._tar_colors[-1] != np.array(transfer_lab.rgb2lab_opencv([255,255,255]))):
            self._tar_colors.append(transfer_lab.rgb2lab_opencv([255,255,255]))
        # if any(self._dom_colors[-1] != np.array(transfer_lab.rgb2lab_opencv([109, 109, 109]))):
        #     self._dom_colors.append(transfer_lab.rgb2lab_opencv([109, 109, 109]))
        #
        # if any(self._tar_colors[-1] != np.array(transfer_lab.rgb2lab_opencv([109, 109, 109]))):
        #     self._tar_colors.append(transfer_lab.rgb2lab_opencv([109, 109, 109]))
        #
        # if any(self._dom_colors[-1] != np.array(transfer_lab.rgb2lab_opencv([149, 149, 149]))):
        #     self._dom_colors.append(transfer_lab.rgb2lab_opencv([149, 149, 149]))
        #
        # if any(self._tar_colors[-1] != np.array(transfer_lab.rgb2lab_opencv([149, 149, 149]))):
        #     self._tar_colors.append(transfer_lab.rgb2lab_opencv([149, 149, 149]))
        src_ab = np.float32(self.get_aff_23(self._dom_colors))
        tar_ab = np.float32(self.get_aff_23(self._tar_colors))

        # ab 变化
        M, status = cv2.findHomography(src_ab, tar_ab, method=cv2.RANSAC, ransacReprojThreshold=15)
        print("status",status.ravel().tolist())
        # 
        mask_status = status.ravel().tolist()
        tar_widgets = []
        for w in (self.layout_widgets(self.tar_color_layout)):
            tar_widgets.append(w)
        for idx, ms in enumerate(mask_status):
            if idx == len(mask_status) - 1:
                continue
            if ms == 0:
                p = tar_widgets[idx].widget().palette()
                r,g,b = transfer_lab.lab2rgb_opencv(self._dom_colors[idx])
                p.setColor(tar_widgets[idx].widget().backgroundRole(),QColor.fromRgb(r,g,b))
                tar_widgets[idx].widget().setPalette(p)
        if M is None:
            return
        # M, inliers = cv2.estimateAffinePartial2D(src_ab, tar_ab, method=cv2.RANSAC)
        # M = cv2.getAffineTransform(src_ab, tar_ab)
        # M, status = cv2.findHomography(src_ab, tar_ab, method=cv2.LMEDScv2.LMEDS,ransacReprojThreshold=1000)
        print(M)

        mode = 2
        img_labsave = cv2.cvtColor(self.cvImage,cv2.COLOR_BGR2LAB)
        hh, ww, _ = img_labsave.shape
        cc = 0
        # print(img_labsave[300,300])
        img_labsave2 = copy.deepcopy(img_labsave)
        img_labsave3 = copy.deepcopy(img_labsave)
        img_labsave2[:, :, 2] = 1
        img_labsave2[:, :, 0], img_labsave2[:, :, 1] = img_labsave[:, :, 1], img_labsave[:, :, 2]
        img_labsave2 = img_labsave2.astype(np.float)
        img_labsave3 = img_labsave3.astype(np.float)
        # img_labsave2=np.reshape(img_labsave2,(-1,3))
        re = np.dot(img_labsave2, M.T)
        re[:, :, 0] /= re[:, :, 2]
        re[:, :, 1] /= re[:, :, 2]

        def out_range(r,g,b):
            return r < 0 or r > 255 or g < 0 or g > 255 or b < 0 or b > 255
        def scale_ab(v):
            sign = 1
            if v < 0:
                sign = -1
            # print(v,math.pow(sign*v,0.9)*sign)
            return math.sqrt(sign*v)*sign

        def non_linear_image(_lab_v):
            global is_out
            is_out = False
            _a = _lab_v[1]
            _b = _lab_v[2]
            r,g,b = transfer_lab.PURE_LAB2RGB(_lab_v)
            if out_range(r,g,b):
                is_out = True

            _a -= 128
            _b -= 128

            _a = scale_ab(_a) + 128
            _b = scale_ab(_b) + 128
            _lab_v[1] = _a
            _lab_v[2] = _b
            # print("HERE")
            return _lab_v
        # iter = 0
        # global is_out
        is_out = False
        # re = np.apply_along_axis(non_linear_image, 2, re)
        # while iter <= 100 and is_out:
        #     re = np.apply_along_axis(non_linear_image, 2, re)
        #     iter += 1
        #     print("itere count", iter)

        img_labsave3[:, :, 1], img_labsave3[:, :, 2] = re[:, :, 0], re[:, :, 1]
        # TODO img_save3中 会存在超出可视范围的点。
        img_labsave = img_labsave3.astype(np.uint8)
        self.reimage_rgb = cv2.cvtColor(img_labsave, cv2.COLOR_LAB2RGB)
        h, w, _bt = self.cvImage.shape
        btv = _bt * w

        self.mQImage = QImage(cv2.cvtColor(img_labsave, cv2.COLOR_LAB2RGB), w, h, btv, QImage.Format_RGB888)
        self.imageView.setPixmap(QPixmap.fromImage(self.mQImage))
        self.imageView.show()
        # QMessageBox.question(self, 'Message', "图片优化计算完成" + str(is_out), QMessageBox.Yes,
        #                      QMessageBox.Yes)



    def choose_color(self, event, source_object=None,index=None):
        # print("clicked from", source_object)
        print("index",index)
        color = QColorDialog.getColor()
        if color.isValid():
            # print(color.name())
            # print(color.getRgb())
            p = source_object.palette()
            p.setColor(source_object.backgroundRole(), color)
            source_object.setPalette(p)

            # 更新目标颜色的ab值
            r,g,b,_=color.getRgb()
            source_object.setStyleSheet("background-color: %s" % (QColor.fromRgb(r, g, b).name()))

            self._tar_colors[index] = transfer_lab.rgb2lab_opencv([r,g,b])

            # Todo 更新当前绘制的图片
            self.update_image()


    @pyqtSlot()
    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open the image that you want to recolor")

        if fileName:
            self.fileName = fileName
            self.cvImage = cv2.imread(self.fileName)
            h,w, _bt= self.cvImage.shape
            btv = _bt*w
            # cv2.imwrite("/home/qiliux/test1.png",self.cvImage)

            self.mQImage  = QImage(cv2.cvtColor(self.cvImage,cv2.COLOR_BGR2RGB),w,h, btv,QImage.Format_RGB888)
            self.imageView.setPixmap(QPixmap.fromImage(self.mQImage))

            # self.find_dom_colors()

    @pyqtSlot()
    def on_click(self):
        buttonReply = QMessageBox.question(self,"PPP","Are you working fun?" + self.textbox.text(),QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        if buttonReply == QMessageBox.Yes:
            print("YES")


class FindDomColorsThread(QThread):
    dom_colors = pyqtSignal( list)

    def __init__(self,image_proc_BGR, parent=None):
        super().__init__(parent)
        self.image_proc_BGR = image_proc_BGR

    def run(self):
        def template_clustering(start_medoids, sample, tolerance=0.25, show=True):

            kmedoids_instance = kmedoids(sample, start_medoids, tolerance);
            (ticks, result) = timedcall(kmedoids_instance.process);

            clusters = kmedoids_instance.get_clusters();
            medoids = kmedoids_instance.get_medoids();
            print("Execution time: ", ticks, "\n");

            if (show is True):
                visualizer = cluster_visualizer(1);
                visualizer.append_clusters(clusters, sample, 0);
                visualizer.append_cluster([sample[index] for index in start_medoids], marker='+', markersize=15);
                visualizer.append_cluster(medoids, marker='*', markersize=5);
                visualizer.show();
            return medoids

        eg_2 = self.image_proc_BGR

        eg_2_lab = cv2.cvtColor(eg_2, cv2.COLOR_BGR2LAB)
        eg_2_hls = cv2.cvtColor(eg_2, cv2.COLOR_BGR2HLS)

        eg_2_lab_3 = eg_2_lab.reshape((-1, 3))
        # 将图片数据 转化为可聚类的 2维特征数据，只取AB 因为我们只在 AB平面变化
        eg_2_ab = eg_2_lab[:, :, 1:].reshape((-1, 2))
        print(eg_2_ab.shape)
        eg_2_ab, inxs = np.unique(eg_2_ab, return_index=True, axis=0)
        print(inxs)
        eg_3_ab = eg_2_hls.reshape((-1, 3))
        eg_3_ab = eg_3_ab[inxs]
        start_idxs = []
        print(eg_2_ab.shape)

        # 根据直方图的Hue范围选择 初始点
        # 分段统计直方图
        # 30 20 10 5
        bin_size = 15
        bins = int(180 / bin_size)
        hist = cv2.calcHist([eg_2_hls], [0], None, [bins], [0, 180])

        for i in range(bins):
            ix = np.isin(eg_3_ab[:, 0], range(i * bin_size, (i + 1) * bin_size))
            # print(ix.shape)
            rows = np.where(ix == True)[0]
            if len(rows) > 0:
                start_idxs.append(rows[0])
        for idx in start_idxs:
            # print(eg_3_ab[idx])
            pass

        # 获得 ab的拓扑结构保持点
        medoid_abs = template_clustering(start_idxs, eg_2_ab.astype(np.float), show=False)
        medoid_abs = np.array(medoid_abs).astype(np.uint8)
        print("聚类",medoid_abs)
        dom_colors = []
        for ab in medoid_abs:
            dom_t = eg_2_lab_3[np.all(eg_2_lab_3[:, 1:] == ab, axis=1)]
            if len(dom_t) > 0:
                dom_colors.append(dom_t[0])
        print(dom_colors)
        self.dom_colors.emit(dom_colors)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App()
    sys.exit(app.exec_())