###########################
# GUI based landmark registration
# Created by Mingzhen Shao  2023/9/2
# SCI UofU
###########################
#################TODO#################


import sys
from PyQt5.QtWidgets import QDialog, QLineEdit,  QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsOpacityEffect, QScrollArea, QVBoxLayout as VBoxLayout

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtCore import Qt, QPoint
import qdarkstyle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.ndimage import map_coordinates

import skimage as ski
from tps import ThinPlateSpline
from concurrent.futures import ThreadPoolExecutor


global sf1234

sf1234=3

#Can set default folder here to make finding files easier->
DEFAULT_FOLDER = "/Users/jbonaventura/Downloads/RabbitData"

class ImageViewer(QDialog):
    def __init__(self, image):
        super().__init__()
        self.initUI(image)
        self.setScaledContents(True)

    def initUI(self, image):
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1800, 1600)

        layout = QVBoxLayout()

        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        self.scene.addItem(pixmap_item)

        layout.addWidget(self.view)

        self.setLayout(layout)


class ImageBlenderDialog(QDialog):
    def __init__(self, img1, img2):
        super().__init__()
        self.setWindowTitle("Image Blender")
        self.setGeometry(100, 100, 2000, 1800)
        
        self.img1 = img1
        self.img2 = img2
        self.opacity = 0.5
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create QGraphicsView widgets to display images
        self.view = QGraphicsView(self)
        self.update_images()

        # Create a horizontal layout for sliders
        slider_layout = QHBoxLayout()

        # Create a slider to control opacity
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        slider_layout.addWidget(self.opacity_slider)

        # Create labels for opacity slider
        slider_label = QLabel("Opacity:")
        self.opacity_value_label = QLabel("50%")
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.opacity_value_label)

        layout.addWidget(self.view)
        layout.addLayout(slider_layout)

        self.setLayout(layout)

    def update_opacity(self):
        self.opacity = self.opacity_slider.value() / 100
        self.update_images()
        self.opacity_value_label.setText(f"{self.opacity_slider.value()}%")

    def update_images(self):
        scene = QGraphicsScene()

        pixmap1 = QPixmap.fromImage(self.img1)
        half_width = pixmap1.width() // sf1234
        half_height = pixmap1.height() // sf1234
        pixmap1 = pixmap1.scaled(half_width, half_height)

        pixmap2 = QPixmap.fromImage(self.img2)
        half_width = pixmap2.width() // sf1234
        half_height = pixmap2.height() // sf1234
        pixmap2 = pixmap2.scaled(half_width, half_height)

        item1 = QGraphicsPixmapItem(pixmap1)
        item2 = QGraphicsPixmapItem(pixmap2)

        # Apply opacity effect to item1 (img1)
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(self.opacity)
        item1.setGraphicsEffect(opacity_effect)

        scene.addItem(item2)  # Add img2
        scene.addItem(item1)  # Add img1 on top

        self.view.setScene(scene)



# def tps_transform(source_points, target_points, lambda_reg=0.01):
#     num_points = source_points.shape[0]
    
#     # Compute the pairwise squared distances between points
#     K = cdist(source_points, source_points, 'euclidean')
#     K = np.where(K == 0, 1e-10, K)  # Avoid division by zero
    
#     # Compute the TPS matrix
#     P = np.column_stack((np.ones((num_points, 1)), source_points))
#     L = np.zeros((num_points + 3, num_points + 3))
#     L[:num_points, :num_points] = K
#     L[:num_points, -3:] = P
#     L[-3:, :num_points] = P.T
    
#     # Regularization term
#     L[:num_points, :num_points] += lambda_reg * np.eye(num_points)
    
#     # Solve the linear system for the TPS parameters
#     V = np.zeros((num_points + 3, 2))
#     V[:num_points, :] = target_points
#     theta = solve(L, V, overwrite_a=True, overwrite_b=True)
    
#     return theta


# def apply_tps_transform(image, source_points, target_points, theta):
#     num_points = source_points.shape[0]

#     print(image.shape)
    
#     # Compute the affine transformation component
#     P = np.column_stack((np.ones((num_points, 1)), source_points))
#     affine = np.dot(P, theta[-3:])
#     # print(affine.shape)
    
#     # Compute the non-linear component
#     distances = cdist(source_points, target_points, 'euclidean')
#     # print((distances**2).shape, (distances**2))
#     # print()
#     # print(theta[:num_points].shape, theta[:num_points])

#     non_linear = np.sum(np.matmul(distances**2, theta[:num_points]), axis=1)
    
#     # Apply the transformation
#     transformed_points = affine + non_linear[:, np.newaxis]
#     transformed_image = np.zeros_like(image)
#     for channel in range(3):
#         transformed_image[:, :, channel] = map_coordinates(image[:, :, channel], (transformed_points[:, 1], transformed_points[:, 0]), order=3, mode='reflect')

#     # transformed_image = map_coordinates(image, (transformed_points[:, 1], transformed_points[:, 0]), order=3)
#     # transformed_image = transformed_image.reshape(image.shape)
    
#     return transformed_image


# Define a function to calculate the TPS transformation
def tps_transform(src_points, dest_points, lambda_param=0.1):
    # Calculate the pairwise Euclidean distances between points
    pairwise_distances = cdist(src_points, src_points)
    
    # Compute the TPS matrix K
    K = pairwise_distances**2 * np.log(pairwise_distances + 1e-6)
    np.fill_diagonal(K, 0)
    
    # Construct the augmented matrix P
    P = np.vstack((np.ones((1, len(src_points))), src_points.T)).T
    
    # Assemble the linear system for the TPS parameters
    L = np.block([[K, P], [P.T, np.zeros((3, 3))]])
    
    # Calculate the target displacements
    target_displacements = np.vstack((dest_points, np.zeros((3, 2))))
    
    # Solve for the TPS parameters
    params = solve(L, target_displacements, overwrite_a=True, overwrite_b=True)
    
    # Extract the affine and non-affine components
    affine = params[-3:, :]
    non_affine = params[:-3, :]

     
    # Define a function to apply the TPS transformation
    def transform_point(point):
        
        affine_component = np.dot(affine.T, np.array([1, point[0], point[1]]))
        # print(affine_component)
        print(non_affine.shape)
        # print(np.log(cdist(np.array([point]), src_points).squeeze()+ 1e-6).shape)

        non_affine_component = np.sum(np.matmul(np.log(cdist(np.array([point]), src_points).squeeze() + 1e-6), non_affine))
        transformed_point = point +  affine_component  #non_affine_component       # +
        return transformed_point
    
    return transform_point



class ImageSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.image1_path = None
        self.image2_path = None
        self.points_image1 = []
        self.points_image2 = []

        self.setWindowTitle("Image Point Selector")
        self.setGeometry(100, 100, 1800, 1000)

        central_widget = QWidget(self)
        
        self.layout = QVBoxLayout()

        image_layout = QHBoxLayout()


        self.label1 = QLabel(self)
        self.label2 = QLabel(self)

        #  # Create a scroll area
        # self.scroll_area_1 = QScrollArea(self)
        # self.scroll_area_1.setWidget(self.label1)
        # self.scroll_area_1.setWidgetResizable(True)

        # self.scroll_area_2 = QScrollArea(self)
        # self.scroll_area_2.setWidget(self.label2)
        # self.scroll_area_2.setWidgetResizable(True)

        # Set a fixed size for the QLabel
        # self.label1.setFixedSize(800, 800)
        # self.label2.setFixedSize(800, 800)



        image_layout.addWidget(self.label1)
        image_layout.addWidget(self.label2)
        # self.layout.addLayout(image_layout)
        
        button_layout = QHBoxLayout()

        self.checkbox = QCheckBox("Loading Annotations (Preconditions required)")
        self.layout.addWidget(self.checkbox)

        self.load_button1 = QPushButton("Block Face Image")
        self.load_button1.clicked.connect(self.load_image1)
        button_layout.addWidget(self.load_button1)
        
        self.load_button2 = QPushButton("Open Histology Image")
        self.load_button2.clicked.connect(self.load_image2)
        button_layout.addWidget(self.load_button2)
        
        self.layout.addLayout(button_layout)

        self.load_button3 = QPushButton("Delete points")
        self.load_button3.clicked.connect(self.delete)
        self.layout.addWidget(self.load_button3)


        method_layout = QHBoxLayout()
        self.apply_button_1 = QPushButton("Apply_rigid")
        self.apply_button_1.clicked.connect(self.rigid)
        method_layout.addWidget(self.apply_button_1)

        self.apply_button_2 = QPushButton("Apply_TP")
        self.apply_button_2.clicked.connect(self.thin_plate)
        method_layout.addWidget(self.apply_button_2)



        self.layout.addLayout(method_layout)


        # self.apply_button = QPushButton("Save")
        # self.apply_button.clicked.connect(self.save)
        # self.layout.addWidget(self.apply_button)

        save_layout = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_landmarks)
        save_layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        save_layout.addWidget(self.save_button)

        self.layout.addLayout(save_layout)

        image_layout.addLayout(self.layout)

     
        central_widget.setLayout(image_layout)

        # # Wrap central_widget in a QScrollArea
        # scroll_area = QScrollArea(self)
        # scroll_area.setWidgetResizable(True)
        # scroll_area.setWidget(central_widget)

        self.setCentralWidget(central_widget)

    def load_image1(self):
        options = QFileDialog.Options()
        self.image1_path, _ = QFileDialog.getOpenFileName(self, "Open Block Face Image", DEFAULT_FOLDER, "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)", options=options)
        pixmap_1 = QPixmap(self.image1_path)

        half_width = pixmap_1.width() // sf1234
        half_height = pixmap_1.height() // sf1234
        pixmap_1 = pixmap_1.scaled(half_width, half_height)

        self.label1.setPixmap(pixmap_1)
        self.label1.setAlignment(Qt.AlignCenter)


        # self.display_image(self.image1_path, self.label1)
        print("size",pixmap_1.size())
        self.label1.setFixedSize((pixmap_1.size()))
        self.points_image1 = []

    def load_image2(self):
        options = QFileDialog.Options()
        self.image2_path, _ = QFileDialog.getOpenFileName(self, "Open Histology Image", DEFAULT_FOLDER, "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)", options=options)

        pixmap_2 = QPixmap(self.image2_path)

        half_width = pixmap_2.width() // sf1234
        half_height = pixmap_2.height() // sf1234
        pixmap_2 = pixmap_2.scaled(half_width, half_height)

        self.label2.setPixmap(pixmap_2)
        self.label2.setAlignment(Qt.AlignCenter)


        # self.display_image(self.image2_path, self.label2)
        self.label2.setFixedSize(pixmap_2.size())
        self.points_image2 = []

    def load_landmarks(self):
        options = QFileDialog.Options()
        _landmarks_file, _ = QFileDialog.getOpenFileName(self, "Load Landmarks *.npy", DEFAULT_FOLDER, "Landmarks (*.npy);;All Files (*)", options=options)
        # print(_landmarks_file)
        _landmarks = np.load(_landmarks_file, allow_pickle=True)
        # print(_landmarks)
        self.points_image1 = _landmarks[0]
        self.points_image2 = _landmarks[1]

        print(self.points_image1)

        self.drawMarks(self.label1, QColor(255,0,0), self.points_image1)
        self.drawMarks(self.label2, QColor(0,255,0), self.points_image2)
        # pass
        
    def save(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save transformed image, *.png", DEFAULT_FOLDER, "Images (*.png);;All Files (*)", options=options)
        if filename:
           
            cv2.imwrite(filename, self.output_image)
            _landmarks = [self.points_image1, self.points_image2]
            np.save(filename.split('.png')[0]+'_landmarks.npy', _landmarks)

            if(self.checkbox.isChecked()):
                cv2.imwrite(filename.split('.png')[0]+'-label.png', self.output_an)

            print(f"Image saved to: {filename}")
            # self.accept()
        else:
            print("Please enter a valid save path.")

    # def display_image(self, image_path, label):
    #     if image_path:
    #         self.pixmap = QPixmap(image_path)
    #         label.setPixmap(self.pixmap)
    #         label.setAlignment(Qt.AlignCenter)

    def rigid(self):
        if len(self.points_image1) == len(self.points_image2):
            # print("Number of points selected on both images is equal.")
            pts_img1 = np.array([[point.x(), point.y()] for point in self.points_image1]) *sf1234
            
            pts_img2 = np.array([[point.x(), point.y()] for point in self.points_image2])*sf1234
            print("Points in image 1:", self.points_image1)
            print("Points in image 2:", pts_img2)
            
            # Call your function F(x) here with self.points_image1 and self.points_image2
          
            # Calculate rigid transformation (translation and rotation)
            M, _ = cv2.estimateAffine2D(pts_img2, pts_img1)

            print("Rigid Transformation Matrix:")
            print(M)

            img1 = cv2.imread(self.image1_path)
            img2 = cv2.imread(self.image2_path)

            h, w, _ = img1.shape
            self.output_image = cv2.warpAffine(img2, M, (w, h))


            height, width, channel = img1.shape
            bytesPerLine = 3 * width
            qImg1 = QImage(img1.data, width, height, bytesPerLine, QImage.Format_BGR888).scaled(width // 2, height// 2, Qt.KeepAspectRatio)

            height, width, channel = self.output_image.shape
            bytesPerLine = 3 * width
            qImg2 = QImage(self.output_image.data, width, height, bytesPerLine, QImage.Format_BGR888).scaled(width // 2, height // 2, Qt.KeepAspectRatio)

            dialog = ImageBlenderDialog(qImg1, qImg2)
            dialog.exec_()

            # image_viewer = ImageViewer(qImg)
            # image_viewer.exec_()


        else:
            print("Number of points selected on both images is not equal.")


    def thin_plate(self):
        if len(self.points_image1) == len(self.points_image2):
            img1 = cv2.imread(self.image1_path)
            _height, _width, _channel = img1.shape
            bytesPerLine = 3 * _width
            qImg1 = QImage(img1.data, _width, _height, bytesPerLine, QImage.Format_BGR888)

            pts_img1 = np.array([[point.x(), point.y()] for point in self.points_image1]) *sf1234
            pts_img2 = np.array([[point.x(), point.y()] for point in self.points_image2]) *sf1234

            img2 = cv2.imread(self.image2_path)

            splines= ski.transform.ThinPlateSplineTransform.from_estimate(pts_img1,pts_img2)

            self.output_image=ski.transform.warp(img2,splines, output_shape=(_height, _width, _channel))
            #normalize and convert to uint8
            self.output_image=(self.output_image/np.max(self.output_image)*255).astype(np.uint8)

            _height, _width, _channel = self.output_image.shape
            bytesPerLine = 3 * _width
            qImg2 = QImage(self.output_image.data, _width, _height, bytesPerLine, QImage.Format_BGR888)

            dialog = ImageBlenderDialog(qImg1, qImg2)
            dialog.exec_()

        else:
            print("\033[31mNumber of points selected on both images is not equal!\033[0m")



    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.label1.underMouse() and self.image1_path:
              
                relative_pos = event.pos() - self.label1.pos()

                # label_width = self.label1.pixmap().width()
                # label_height = self.label1.pixmap().height()

                self.points_image1.append(relative_pos)

                # print(event.pos())
                print(relative_pos)
              
                # print("label1:", self.label1.pos())
                # print("*************")

                painter1 = QPainter(self.label1.pixmap())
                red_color = QColor(255, 0, 0)
                painter1.setPen(Qt.NoPen)
                painter1.setBrush(red_color)
                painter1.drawEllipse(relative_pos, 5, 5)

                painter1.end()
                
                self.update()


            elif self.label2.underMouse() and self.image2_path:
                
                relative_pos = event.pos() - self.label2.pos()
                self.points_image2.append(relative_pos)

                painter2 = QPainter(self.label2.pixmap())
                green_color = QColor(0, 255, 0)
                painter2.setPen(Qt.NoPen)
                painter2.setBrush(green_color)
                painter2.drawEllipse(relative_pos, 5, 5)
                painter2.end()

                self.update()
        # print(self.points_image1, self.points_image2)

    def drawMarks(self, label, color, landmarks):
        painter = QPainter(label.pixmap())
       
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        for item in landmarks:
            painter.drawEllipse(item, 5, 5)
        painter.end()

        self.update()

    def delete(self):
        # Remove the last selected point on both images
        if len(self.points_image1) == len(self.points_image2):
            if self.points_image1:
                self.points_image1.pop()
                self.update()
            if self.points_image2:
                self.points_image2.pop()
                self.update()
        elif (len(self.points_image1) > len(self.points_image2)):
            self.points_image1.pop()
            self.update()
        elif (len(self.points_image1) < len(self.points_image2)):
            self.points_image2.pop()
            self.update()

        pixmap_1 = QPixmap(self.image1_path)
        half_width = pixmap_1.width() // sf1234
        half_height = pixmap_1.height() // sf1234
        pixmap_1 = pixmap_1.scaled(half_width, half_height)
        self.label1.setPixmap(pixmap_1)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setFixedSize(pixmap_1.size())
        self.drawMarks(self.label1, QColor(255,0,0), self.points_image1)

        pixmap_2 = QPixmap(self.image2_path)
        half_width = pixmap_2.width() // sf1234
        half_height = pixmap_2.height() // sf1234
        pixmap_2 = pixmap_2.scaled(half_width, half_height)
        self.label2.setPixmap(pixmap_2)
        self.label2.setAlignment(Qt.AlignCenter)
        # self.display_image(self.image2_path, self.label2)
        self.label2.setFixedSize(pixmap_2.size())
        self.drawMarks(self.label2, QColor(0,255,0), self.points_image2)
        
        print(self.points_image1, self.points_image2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSelector()
    window.show()
    sys.exit(app.exec_())
