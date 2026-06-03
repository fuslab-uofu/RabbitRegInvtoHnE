import sys
import os
import re
import numpy as np
import nibabel as nib
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QCheckBox, QLabel, QGroupBox, QComboBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob


class TillingGUI(QMainWindow):
    def __init__(self, hne_base_dir, rabbit_mr_root, bf_cropped_dir):
        super().__init__()
        self.setWindowTitle("Multi-Modal Alignment & Tiling Validator")
        self.setGeometry(100, 100, 1200, 800)
        self.hne_base_dir = hne_base_dir
        self.rabbit_mr_root = rabbit_mr_root
        self.bf_cropped_dir = bf_cropped_dir
        self.tilesize = 100
        self.nifti_cache = {}

        self.left_asset_path = None
        self.right_asset_path = None

        self.discover_all_volumes()
        self.load_histology_data()
        self.init_ui()



    def discover_all_volumes(self):
        """
        Recursively scans the MR root folder for any Nifti files residing
        within a 'RegDataOut' directory sub-tree.
        """
        self.available_volumes = [{'display_name': 'H&E Histology', 'path': 'HNE'}]

        # Recursive glob search for all .nii.gz files inside the rabbit's MR tree
        search_pattern = os.path.join(self.rabbit_mr_root, '**', '*.nii.gz')
        all_niftis = glob.glob(search_pattern, recursive=True)

        for fpath in sorted(all_niftis):
            # Check if 'RegDataOut' is part of the file path steps
            path_segments = fpath.split(os.sep)
            if 'RegDataOut' in path_segments:
                # Extract context labels from path to make dropdown item clearly distinguishable
                # e.g., 'InVivo_MR' vs 'ExVivo_MR/Block04Out'
                try:
                    reg_index = path_segments.index('RegDataOut')
                    context_label = "/".join(path_segments[reg_index - 2:reg_index])
                except (ValueError, IndexError):
                    context_label = "Unknown Stage"

                display_name = f"[{context_label}] {os.path.basename(fpath)}"

                self.available_volumes.append({
                    'display_name': display_name,
                    'path': fpath
                })

    def load_volume(self, path):
        if path == 'HNE':
            return self.reg_HnE_arr

        if path != 'HNE' and path not in self.nifti_cache:
            reg_volume = nib.load(path)
            volume_data = reg_volume.get_fdata()
            if reg_volume.affine[0, 0] > 0 and reg_volume.affine[1, 1] > 0:
                volume_data = volume_data[::-1, ::-1, :]

            self.nifti_cache[path] = volume_data

        return self.nifti_cache[path]

    def load_histology_data(self):
        """Pre-loads the reference histology slices."""
        reg_HnE_dir = os.path.join(self.hne_base_dir, 'reg')
        self.hne_filenames = sorted(
            f for f in os.listdir(reg_HnE_dir) if f.endswith('.png') and not f.startswith('._'))
        hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in self.hne_filenames]
        self.reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)
        self.num_slices = self.reg_HnE_arr.shape[2]
        hne_parts = self.hne_base_dir.split(os.sep)
        # Dynamic fallback parameters if directory tree sizes change slightly
        self.rabbit_id = hne_parts[10] if len(hne_parts) > 10 else "Unknown"
        self.block = hne_parts[12] if len(hne_parts) > 12 else "Unknown"

    def tiling_tool(self, twoDIm, tile_size):
        rgbmean = np.mean(twoDIm, axis=2)
        whiteIm = np.where(rgbmean > 210, 0, 1)
        twoDIm = twoDIm * whiteIm[:, :, np.newaxis]
        yrem = twoDIm.shape[0] % tile_size
        if yrem:
            ystart = yrem // 2
            ycount = int(twoDIm.shape[0] // tile_size)
        else:
            ycount = int(twoDIm.shape[0] / tile_size)
            ystart = 0
        xrem = twoDIm.shape[1] % tile_size
        if xrem:
            xstart = xrem // 2
            xcount = int(twoDIm.shape[1] // tile_size)
        else:
            xcount = int(twoDIm.shape[1] / tile_size)
            xstart = 0

        tilesList = []
        for row in range(ycount):
            for col in range(xcount):
                tile = twoDIm[
                    ystart + row * tile_size:ystart + (row + 1) * tile_size, xstart + col * tile_size:xstart + (
                                col + 1) * tile_size, :]
                zero_mask = np.all(tile == 0, axis=-1)  # shape: (H, W) boolean
                zero_count = np.sum(zero_mask)
                if zero_count < (tile_size ** 2) / 6:
                    org = [ystart + row * tile_size, xstart + col * tile_size]
                    tilesList.append(org)
        return np.asarray(tilesList)

    def get_bf_slice_index(self, img_number):
        all_files = sorted(f for f in os.listdir(self.bf_cropped_dir) if f.endswith('.tiff') and not f.startswith('._'))
        try:
            match = next(f for f in all_files if img_number in f and f.endswith('_scatter.tiff'))
            return all_files.index(match)
        except StopIteration:
            return 0

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        controls_box = QGroupBox("Cross-Stage Multi-Volume Selector")
        controls_layout = QHBoxLayout()

        dropdown_layout = QVBoxLayout()

        left_label = QLabel("Left Display:")
        self.left_dropdown = QComboBox()

        right_label = QLabel("Right Display:")
        self.right_dropdown = QComboBox()

        self.left_dropdown.addItem("--Select--", None)
        self.right_dropdown.addItem("--Select--", None)

        # Populate UI selector with distinct path classifications
        for vol_info in self.available_volumes:
            self.left_dropdown.addItem(vol_info['display_name'], vol_info['path'])
            self.right_dropdown.addItem(vol_info['display_name'], vol_info['path'])

        self.left_dropdown.setCurrentIndex(0)
        self.right_dropdown.setCurrentIndex(0)
        self.left_dropdown.currentIndexChanged.connect(self.on_left_volume_changed)
        self.right_dropdown.currentIndexChanged.connect(self.on_right_volume_changed)

        dropdown_layout.addWidget(left_label)
        dropdown_layout.addWidget(self.left_dropdown, stretch=1)
        dropdown_layout.addWidget(right_label)
        dropdown_layout.addWidget(self.right_dropdown, stretch=1)


        controls_layout.addLayout(dropdown_layout)

        slider_layout = QHBoxLayout()

        self.slice_label = QLabel(f"Slice: 1 / {self.num_slices}")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_slices - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slice_changed)

        self.grid_checkbox = QCheckBox("Overlay Tiling Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.update_plots)

        slider_layout.addWidget(self.slider, stretch=2)
        slider_layout.addWidget(self.grid_checkbox)
        slider_layout.addWidget(self.slice_label)
        controls_layout.addLayout(slider_layout)

        controls_box.setLayout(controls_layout)
        main_layout.addWidget(controls_box)

    def on_left_volume_changed(self, index):
        self.left_asset_path = self.left_dropdown.itemData(index)
        self.update_plots()

    def on_right_volume_changed(self, index):
        self.right_asset_path = self.right_dropdown.itemData(index)
        self.update_plots()

    def on_slice_changed(self, value):
        self.slice_label.setText(f"Slice: {value + 1} / {self.num_slices}")
        self.update_plots()

    def draw_panel_data(self, ax, path, hne, img_number):
        if path == 'HNE':
            ax.imshow(hne)
            ax.set_title(f"H&E Histology (Img: {img_number})")
        else:
            vol_arr = self.load_volume(path)
            slice_num = self.get_bf_slice_index(img_number)

            if slice_num < vol_arr.shape[2]:
                MR_Slice = vol_arr[:, :, slice_num].T
            else:
                MR_Slice = np.zeros((hne.shape[0] // 4, hne.shape[1] // 4))

            vol_upsampled = np.repeat(np.repeat(MR_Slice, 4, axis=0), 4, axis=1)
            MR_Slice_us = vol_upsampled[:hne.shape[0], :hne.shape[1]]

            ax.imshow(MR_Slice_us, cmap='gray')
            ax.set_title(f"MR Matrix: {os.path.basename(path)} (Slice: {slice_num})")

    def update_plots(self):
        idx = self.slider.value()

        self.axes[0].clear()
        self.axes[1].clear()

        hne_ds_im = self.reg_HnE_arr[:, :, idx, :]
        img_number = re.search(r'\d+', self.hne_filenames[idx]).group()

        if self.left_asset_path is not None:
            self.draw_panel_data(self.axes[0], self.left_asset_path, hne_ds_im, img_number)
        else:
            self.axes[0].text(0.5, 0.5, "No Asset Selected", color="gray", ha="center", va="center")
            self.axes[0].set_title("Left View: Empty")

            # Render Right Panel if something is selected, otherwise leave dark/empty
        if self.right_asset_path is not None:
            self.draw_panel_data(self.axes[1], self.right_asset_path, hne_ds_im, img_number)
        else:
            self.axes[1].text(0.5, 0.5, "No Asset Selected", color="gray", ha="center", va="center")
            self.axes[1].set_title("Right View: Empty")

        if self.grid_checkbox.isChecked():
            origin_list = self.tiling_tool(hne_ds_im, self.tilesize)

            if origin_list.ndim == 2 and len(origin_list) > 0:
                for q in range(len(origin_list)):
                    row, col = origin_list[q, 0], origin_list[q, 1]

                    rect = patches.Rectangle((col, row), self.tilesize, self.tilesize,
                                                 linewidth=1, edgecolor='cyan', facecolor='none')
                    self.axes[0].add_patch(rect)
                    rect2 = patches.Rectangle((col, row), self.tilesize, self.tilesize,
                                             linewidth=1, edgecolor='cyan', facecolor='none')
                    self.axes[1].add_patch(rect2)

        for ax in self.axes:
            ax.axis('off')
        self.figure.tight_layout(pad=1.5)
        self.figure.subplots_adjust(wspace=0.25)
        self.canvas.draw()

RabbitFolder='/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID="R23-292"
Block = 4

if __name__ == '__main__':
    blockId = "Block" + f"{Block:02d}"

    # POINT THIS TO THE FOLDER CONTAINING INTERM EDIATE NII.GZ VOLUMES
    # ROOT folder that branches into InVivo_MR/, ExVivo_MR/, etc.
    rabbit_mr_root = os.path.join(RabbitFolder, RabbitID)
    hne_base = os.path.join(rabbit_mr_root, 'HnE', blockId)

    rabbase = os.path.split(os.path.split(hne_base)[0])[0]
    bf_cropped = os.path.join(rabbase, 'BlockFace_RGB', blockId, 'CroppedImages')

    app = QApplication(sys.argv)
    viewer = TillingGUI(hne_base, rabbit_mr_root, bf_cropped)
    viewer.show()
    sys.exit(app.exec_())
