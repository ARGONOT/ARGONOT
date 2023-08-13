import subprocess
subprocess.check_call(['python', '-m', 'ensurepip', '--upgrade'])
subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])

import unicodedata  
import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import shutil
from PyQt5.QtCore import Qt


sco_state = ["Skolyoz yok","Skolyoz Eşiği Altında","Tehlikesiz","\n Omurganın 10 derece ve üstü eğriliklerde skolyoz olarak kabul edilir.\n Danışan kişinin herhangi bir skolyozu mevcut değildir.\n Yine de daha profesyonel bir destek almak için Ortopedi bölümüne başvurulması önerilir.\n\n \t\t Sağlıklı günler dileriz. "]



class Ui_MainWindow(object):    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        MainWindow.setMaximumSize(QtCore.QSize(640, 500))
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        MainWindow.setFont(font)
        MainWindow.setMouseTracking(False)
        MainWindow.setTabletTracking(False)
        MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(170, 0, 262, 80))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.skolyoztesti = QtWidgets.QLabel(self.gridLayoutWidget)
        self.skolyoztesti.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.skolyoztesti.setFont(font)
        self.skolyoztesti.setObjectName("skolyoztesti")
        self.gridLayout.addWidget(self.skolyoztesti, 0, 0, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(380, 350, 250, 80))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.hospital = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.hospital.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.hospital.setFont(font)
        self.hospital.setObjectName("hospital")
        self.gridLayout_2.addWidget(self.hospital, 0, 0, 1, 1)
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(250, 260, 141, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.start.setFont(font)
        self.start.setObjectName("start")
        self.start.clicked.connect(self.started)
        self.path = QtWidgets.QLineEdit(self.centralwidget)
        self.path.setGeometry(QtCore.QRect(100, 200, 351, 21))
        self.path.setObjectName("path")
        self.path.setText("Skolyoz röntgeni seçip testi başlatınız.")
        self.path.setEnabled(False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def setPath(self):
        file_dialog = QtWidgets.QFileDialog(self.centralwidget)
        file_path, _ = file_dialog.getOpenFileName(self.centralwidget, "Röntgen Seç", "", "Tüm Dosyalar (*.*)")
        if file_path:
            self.path.setText(file_path)
            return file_path

    def started(self):
        try:
            for root, dirs, files in os.walk(os.getcwd()):
                for file in files:
                    if file == 'detect.detect':
                        exp_file = os.path.join(root, file)
                        exp_file = exp_file.split("detect.detect")[0] + "exp"
                        print(exp_file)
            shutil.rmtree(exp_file)
        except: 
            pass

        coordinates = []
        def custom_circle(x, img, color=None, label=None, line_thickness=3):
            tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
            color = color or [random.randint(0, 255) for _ in range(3)]
            x_circle = int((x[0]+x[2])/2)
            y_circle = int((x[1]+x[3])/2)
            center_circle = (x_circle,y_circle)
            cv2.circle(img, center_circle, 1 ,color, thickness=6)
            coordinates.append(center_circle)
            
        def calculate_angle(x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            angle = math.atan2(dy, dx)
            angle_degrees = math.degrees(angle)
            return angle_degrees

        def angle(coordinates,n, img):
            sco_degree = 0
            th1 = 10
            th2 = 10
            control_param = True
            control_param_2 = True
            if(len(coordinates) == n):
                coordinates = sorted(coordinates, key=lambda coord: coord[1])
                max_cobb = max(coordinates, key=lambda coord: coord[0])
                max_index = coordinates.index(max_cobb)
                start_coordinate = coordinates[0]
                finish_coordinate = coordinates[n-1]
                for index,coordinate in enumerate(coordinates): 
                    if(control_param == True):
                        if(index <= max_index):
                            kirilma_1 = abs(coordinate[0] - start_coordinate[0])
                            if(kirilma_1 > th1):
                                c1 = coordinate
                                print(c1)
                                cv2.line(img, c1, max_cobb, (255, 0, 0), thickness=10)
                                angle1 = calculate_angle(c1[0],c1[0],max_cobb[0],max_cobb[1])
                                print("Angle 1 : ",angle1)
                                control_param = False
                            
                    else:
                        if(control_param_2 == True):
                            if(index > max_index):
                                kirilma_2 = abs(coordinate[0] - finish_coordinate[0])
                                if(kirilma_2 < th2):
                                    c2 = coordinate
                                    print(c2)
                                    cv2.line(img, max_cobb, c2, (255, 0, 0), thickness=10)
                                    angle2 = calculate_angle(max_cobb[0],max_cobb[0],c2[0],c2[1])
                                    print("Angle 2 : ",angle2)
                                    control_param_2 = False
                                    print(-abs(angle1) + abs(angle2))                                 
                                    sco_degree = -abs(angle1) + abs(angle2)
                                    kalan = sco_degree % 1
                                    vg = round(kalan,2)
                                    sco_degree = int(sco_degree) + vg
                                    
                                    if(sco_degree < 10):
                                        sco_state[0] = "Skolyoz yok"
                                        sco_state[1] = sco_degree
                                        sco_state[2] = "Tehlikesiz"
                                        sco_state[3] = "\n Omurganın 10 derece ve üstü eğriliklerde skolyoz olarak kabul edilir.\n Danışan kişinin herhangi bir skolyozu mevcut değildir.\n Yine de daha profesyonel bir destek almak için Ortopedi bölümüne başvurulması önerilir.\n\n \t\t Sağlıklı günler dileriz. "
                                    if(10 <= sco_degree < 20):
                                        sco_state[0] = "Skolyoz var"
                                        sco_state[1] = sco_degree
                                        sco_state[2] = "Biraz Tehlikeli"
                                        sco_state[3] = "\n Danışan kişinin {0} derece skolyozu mevcuttur. \n Danışan kişinin tedaviye ihtiyacı yoktur \n ancak spor yapmaya özen göstermesi skolyozun ileri derecelere gelmemesi için gereklidir. \n Yine de daha profesyonel bir destek almak için Ortopedi bölümüne başvurulması önerilir. \n\n \t\t Sağlıklı günler dileriz.".format(sco_degree)
                                        
                                    if(20 <= sco_degree < 40):
                                        sco_state[0] = "Skolyoz var"
                                        sco_state[1] = sco_degree
                                        sco_state[2] = "Tehlikeli"
                                        sco_state[3] = "\n Danışan kişinin {0} derece skolyozu mevcuttur. \n Danışan kişinin tedaviye ihtiyacı vardır.  \n Profesyonel bir destek alabilmesi için en yakın zamanda Ortopedi bölümüne başvurması önerilir.\n\n \t\t Sağlıklı günler dileriz. ".format(sco_degree)
                                            
                                    if(40 <= sco_degree):
                                        sco_state[0] = "Skolyoz var"
                                        sco_state[1] = sco_degree
                                        sco_state[2] = "Çok Tehlikeli"
                                        sco_state[3] = "\n Danışan kişinin {0} derecede ciddi bir skolyozu mevcuttur. \n Danışan kişinin tedavi yanında ameliyat ihtiyacı olabilir. \n Profesyonel bir destek alabilmesi için en yakın zamanda Ortopedi bölümüne başvurması önerilir.\n\n \t\t Sağlıklı günler dileriz.".format(sco_degree)
                                         
    
                                                     

        def detect(save_img=False):
            source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
            save_img = not opt.nosave and not source.endswith('.txt') 
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))

            
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

            
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu' 

            
            model = attempt_load(weights, map_location=device)  
            stride = int(model.stride.max()) 
            imgsz = check_img_size(imgsz, s=stride)  

            if trace:
                model = TracedModel(model, device, opt.img_size)

            if half:
                model.half()  

    
            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

            
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  
                img /= 255.0  
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]

                
                t1 = time_synchronized()
                with torch.no_grad():   
                    pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                
                for i, det in enumerate(pred):  
                    if webcam:  
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  
                    save_path = str(save_dir / p.name)  
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
                    if len(det):
                    
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  


                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                custom_circle(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                angle(coordinates,n, im0)
                                

                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  

                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                            print(f" The image with the result is saved in: {save_path}")
                        else:  
                            if vid_path != save_path:  
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  
                                if vid_cap:  
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)

            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

            print(f'Done. ({time.time() - t0:.3f}s)')

        if __name__ == '__main__':
            parser = argparse.ArgumentParser()
            filepath = self.setPath()
            print(filepath)
            for root, dirs, files in os.walk(os.getcwd()):
                for file in files:
                    if file == 'sco.pt':
                        weight_file = os.path.join(root, file)
            parser.add_argument('--weights', nargs='+', type=str, default=weight_file, help='model.pt path(s)')
            parser.add_argument('--source', type=str, default=filepath, help='source')  
            parser.add_argument('--img-size', type=int, default=800, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default='runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
            opt = parser.parse_args()
            print(opt)
            detect()
            
            
            for root, dirs, files in os.walk(os.getcwd()):
                for file in files:
                    if file == 'detect.detect':
                        angle_image = os.path.join(root, file)
                        angle_image = angle_image.split("detect.detect")[0] + "exp\\"
                        ang_f =os.listdir(angle_image)[0]
                        angle_image = angle_image + ang_f
                        angle_image = angle_image.replace("\\", "/")
                        
            self.Form = QtWidgets.QWidget()
            class Ui_Form(object):
                def setupUi(self, Form):
                    Form.setObjectName("Form")
                    Form.resize(1040, 740)
                    Form.setMinimumSize(QtCore.QSize(1040, 740))
                    Form.setMaximumSize(QtCore.QSize(1040, 740))
                    self.gridLayoutWidget = QtWidgets.QWidget(Form)
                    self.gridLayoutWidget.setGeometry(QtCore.QRect(-10, 0, 391, 431))
                    self.gridLayoutWidget.setObjectName("gridLayoutWidget")
                    self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
                    self.gridLayout.setContentsMargins(0, 0, 0, 0)
                    self.gridLayout.setObjectName("gridLayout")
                    self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
                    self.label_2.setStyleSheet("image: url("+filepath+");")
                    self.label_2.setText("")
                    self.label_2.setObjectName("label_2")
                    self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
                    font = QtGui.QFont()
                    font.setPointSize(20)
                    font.setBold(True)
                    font.setWeight(75)
                    self.lineEdit = QtWidgets.QTextEdit(Form)
                    self.lineEdit.setGeometry(QtCore.QRect(220, 460, 561, 271))
                    self.lineEdit.setObjectName("lineEdit")
                    self.lineEdit.setText(str(sco_state[3]))
                    self.lineEdit.setAlignment(Qt.AlignCenter)
                    self.lineEdit.setEnabled(False)
                    self.lineEdit_2 = QtWidgets.QLineEdit(Form)
                    self.lineEdit_2.setGeometry(QtCore.QRect(420, 280, 171, 41))
                    self.lineEdit_2.setObjectName("lineEdit_2")
                    self.lineEdit_2.setText(str(sco_state[1]))
                    self.lineEdit_2.setAlignment(Qt.AlignCenter)
                    self.lineEdit_2.setEnabled(False)
                    self.label_3 = QtWidgets.QLabel(Form)
                    self.label_3.setGeometry(QtCore.QRect(450, 250, 101, 31))
                    self.label_3.setObjectName("label_3")
                    self.lineEdit_3 = QtWidgets.QLineEdit(Form)
                    self.lineEdit_3.setGeometry(QtCore.QRect(420, 380, 171, 41))
                    self.lineEdit_3.setObjectName("lineEdit_3")
                    self.lineEdit_3.setText(str(sco_state[2]))
                    self.lineEdit_3.setAlignment(Qt.AlignCenter)
                    self.lineEdit_3.setEnabled(False)
                    self.label_4 = QtWidgets.QLabel(Form)
                    self.label_4.setGeometry(QtCore.QRect(450, 350, 121, 31))
                    self.label_4.setObjectName("label_4")
                    self.gridLayoutWidget_2 = QtWidgets.QWidget(Form)
                    self.gridLayoutWidget_2.setGeometry(QtCore.QRect(650, 0, 391, 431))
                    self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
                    self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
                    self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
                    self.gridLayout_2.setObjectName("gridLayout_2")
                    self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_2)
                    self.label_5.setStyleSheet("image: url("+angle_image+");")
                    self.label_5.setText("")
                    self.label_5.setObjectName("label_5")
                    self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
                    self.label = QtWidgets.QLabel(Form)
                    self.label.setGeometry(QtCore.QRect(390, -10, 290, 90))
                    self.label.setObjectName("label")
                    self.lineEdit_4 = QtWidgets.QLineEdit(Form)
                    self.lineEdit_4.setGeometry(QtCore.QRect(420, 180, 171, 41))
                    self.lineEdit_4.setObjectName("lineEdit_4")
                    self.lineEdit_4.setText(str(sco_state[0]))
                    self.lineEdit_4.setAlignment(Qt.AlignCenter)
                    self.lineEdit_4.setEnabled(False)
                    self.label_6 = QtWidgets.QLabel(Form)
                    self.label_6.setGeometry(QtCore.QRect(450, 150, 121, 31))
                    self.label_6.setObjectName("label_6")

                    self.retranslateUi(Form)
                    QtCore.QMetaObject.connectSlotsByName(Form)

                def retranslateUi(self, Form):
                    _translate = QtCore.QCoreApplication.translate
                    Form.setWindowTitle(_translate("Form", "Form"))
                    self.label_3.setText(_translate("Form", "Skolyoz Acısı"))
                    self.label_4.setText(_translate("Form", "Tehlike Durumu"))
                    self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600; color:#cc0000;\">SKOLYOZ SONUÇLARI</span></p></body></html>"))
                    self.label_6.setText(_translate("Form", "Skolyoz Durumu"))
            
            self.ui = Ui_Form()
            self.ui.setupUi(self.Form)
            self.Form.show()
   

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SKOLYOZ TESTİ"))
        self.skolyoztesti.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:26pt; color:#204a87;\">SKOLYOZ TESTİ</span></p></body></html>"))
        self.hospital.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#cc0000;\">GÖKÇE HOSPITAL</span></p></body></html>"))
        self.start.setText(_translate("MainWindow", "Testi Başlat"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
