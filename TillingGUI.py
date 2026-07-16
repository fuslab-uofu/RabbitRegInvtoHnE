import sys
import os
import re
import numpy as np
import nibabel as nib
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QCheckBox, QLabel, QGroupBox, QComboBox, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import glob
from SADMS import relaxation

T1_RANGE = (0, 3000)
T2_RANGE = (0, 200)


def get_mr_style(filename):
    name = filename.lower()
    if 'temp' in name:
        return {'kind': 'cmap', 'cmap': 'hot'}
    if 'map' in name and 't1' in name:
        return {'kind': 'relaxation', 'maptype': 'T1', 'loLev': T1_RANGE[0], 'upLev': T1_RANGE[1]}
    if 'map' in name and 't2' in name:
        return {'kind': 'relaxation', 'maptype': 'T2', 'loLev': T2_RANGE[0], 'upLev': T2_RANGE[1]}
    return {'kind': 'cmap', 'cmap': 'gray'}


class TillingGUI(QMainWindow):
    def __init__(self, hne_base_dir, rabbit_mr_root, bf_cropped_dir, block_id):
        super().__init__()
        self.setWindowTitle("Multi-Modal Alignment & Tiling Validator")
        self.setGeometry(100, 100, 1200, 800)
        self.hne_base_dir = hne_base_dir
        self.rabbit_mr_root = rabbit_mr_root
        self.bf_cropped_dir = bf_cropped_dir
        self.block_id = block_id
        self.tilesize = 100
        self.nifti_cache = {}

        self.left_asset_path = None
        self.right_asset_path = None

        self.discover_all_volumes()
        self.load_histology_data()
        self.init_ui()



    def discover_all_volumes(self):
        """
        Scans the MR root folder for the newest RegToBlock registration per stage.
        Only picks up files matching *RegTo{block_id}* inside a RegDataOut folder.
        """
        self.available_volumes = [
            {'display_name': "H&E Histology", 'path': 'HNE'},
            {'display_name': "Blockface RGB", 'path': 'BF'}
        ]

        search_pattern = os.path.join(self.rabbit_mr_root, '**', '*.nii.gz')
        all_niftis = glob.glob(search_pattern, recursive=True)

        grouped_files = {}
        timestamp_regex = re.compile(r'_(\d{2})(\d{2})-(\d{2})(\d{2})\.nii\.gz$')
        block_filter = f'RegTo{self.block_id}'

        for fpath in sorted(all_niftis):
            filename = os.path.basename(fpath)
            if filename.startswith('._'):
                continue
            path_segments = fpath.split(os.sep)
            if 'RegDataOut' not in path_segments:
                continue
            if block_filter not in filename:
                continue

            match = timestamp_regex.search(filename)
            if match:
                month, day, hour, minute = map(int, match.groups())
                time_key = (month, day, hour, minute)
                base_name = filename[:match.start()]
            else:
                time_key = (0, 0, 0, 0)
                base_name = filename

            if base_name not in grouped_files:
                grouped_files[base_name] = []
            grouped_files[base_name].append((time_key, fpath))

        for base_name, occurrences in grouped_files.items():
            occurrences.sort(key=lambda x: x[0])
            newest_fpath = occurrences[-1][1]

            path_segments = newest_fpath.split(os.sep)
            try:
                reg_index = path_segments.index('RegDataOut')
                context_label = "/".join(path_segments[reg_index - 2:reg_index])
            except (ValueError, IndexError):
                context_label = "Unknown Stage"

            display_name = f"[{context_label}] {os.path.basename(newest_fpath)}"
            self.available_volumes.append({
                'display_name': display_name,
                'path': newest_fpath,
            })

    def add_custom_volume(self):
        """Opens a file dialog to manually append a specific .nii.gz volume."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom NIfTI Volume",
            self.rabbit_mr_root,  # Starts browsing inside the current rabbit folder
            "NIfTI Compressed Volumes (*.nii.gz)",  # Strict file filter constraint
            options=options
        )

        # If the user selected a valid file
        if file_path:
            filename = os.path.basename(file_path)
            display_name = f"[Manual Load] {filename}"

            # 1. Append to our asset array track
            new_asset = {
                'display_name': display_name,
                'path': file_path
            }
            self.available_volumes.append(new_asset)

            # 2. Inject directly into the UI comboboxes dynamically
            self.left_dropdown.addItem(display_name, file_path)
            self.right_dropdown.addItem(display_name, file_path)

            # 3. Automatically switch the Right Display to show the newly added file
            new_index = self.right_dropdown.count() - 1
            self.right_dropdown.setCurrentIndex(new_index)

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
        try:
            match_folder = next(
                d for d in os.listdir(self.hne_base_dir)
                if d.lower() == "registered" and os.path.isdir(os.path.join(self.hne_base_dir, d))
            )
            reg_HnE_dir = os.path.join(self.hne_base_dir, match_folder)
        except StopIteration:
            raise FileNotFoundError("Reg folder not found")
        self.hne_filenames = sorted(
            f for f in os.listdir(reg_HnE_dir) if f.endswith('.png') and not f.startswith('._'))
        hne_images = [np.array(Image.open(os.path.join(reg_HnE_dir, f))) for f in self.hne_filenames]
        self.reg_HnE_arr = np.stack(hne_images, axis=2)  # (H, W, N_slices, 3)
        self.num_slices = self.reg_HnE_arr.shape[2]
        hne_parts = self.hne_base_dir.split(os.sep)
        # Dynamic fallback parameters if directory tree sizes change slightly
        self.rabbit_id = hne_parts[10] if len(hne_parts) > 10 else "Unknown"
        self.block = hne_parts[12] if len(hne_parts) > 12 else "Unknown"

    def load_blockface_slice(self, img_number):
        """
        Locates and loads the specific blockface tiff array corresponding to
        the active H&E image index sequence padding requirement.
        """
        try:
            # Gather all tiff images inside the target blockface path
            all_files = sorted(
                f for f in os.listdir(self.bf_cropped_dir) if f.endswith('.tiff') and not f.startswith('._'))

            # Find the specific target filename that contains your current H&E index string
            # e.g., looks for "IMG_0003_scatter.tiff" if img_number is "0003"
            match = next(f for f in all_files if img_number in f and f.endswith('_scatter.tiff'))

            # Construct the absolute path and open the image asset
            bf_path = os.path.join(self.bf_cropped_dir, match)
            return np.array(Image.open(bf_path))

        except (StopIteration, FileNotFoundError):
            # Fallback if a specific slice is missing or index matching fails cleanly
            return None

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

        # Overlay is the most useful view, so it gets a larger panel on the
        # right; the two asset panels are smaller and stacked vertically
        # on the left.
        self.figure = plt.figure(figsize=(14, 8))
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1, 2])
        left_ax = self.figure.add_subplot(grid[0, 0])
        right_ax = self.figure.add_subplot(grid[1, 0])
        overlay_ax = self.figure.add_subplot(grid[:, 1])
        self.axes = [overlay_ax, left_ax, right_ax]
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
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

        # Fixed-width label so its size hint doesn't change with text length --
        # otherwise Qt redistributes space among the row's stretchable widgets
        # (including the slice slider) every time the percentage text changes,
        # making the slice slider visibly resize/shift on every opacity tick.
        self.opacity_label = QLabel("Overlay Opacity: 50%")
        self.opacity_label.setFixedWidth(150)

        # Only three blend levels are useful for eyeballing alignment, and
        # restricting to them means dragging only ever fires 3 redraws instead
        # of up to 100.
        self.opacity_levels = [0.0, 0.5, 1.0]
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(len(self.opacity_levels) - 1)
        self.opacity_slider.setValue(1)
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.setTickInterval(1)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)

        self.upload_btn = QPushButton("📂 Load Custom Volume")
        self.upload_btn.clicked.connect(self.add_custom_volume)

        slider_layout.addWidget(self.slice_label)
        slider_layout.addWidget(self.slider, stretch=2)
        slider_layout.addWidget(self.grid_checkbox)
        slider_layout.addWidget(self.opacity_label)
        slider_layout.addWidget(self.opacity_slider, stretch=1)
        slider_layout.addWidget(self.upload_btn)
        controls_layout.addLayout(slider_layout)

        controls_box.setLayout(controls_layout)
        main_layout.addWidget(controls_box)

    def _show_colorbar_figure(self, path):
        """Opens a standalone colorbar figure for T1/T2 maps, easy to copy/paste."""
        if path in (None, 'HNE', 'BF'):
            return
        style = get_mr_style(os.path.basename(path))
        if style['kind'] != 'relaxation':
            return
        lo, hi = style['loLev'], style['upLev']
        _, lut = relaxation(style['maptype'], np.array([lo, hi]), lo, hi)
        cmap = ListedColormap(lut)
        label = f"{style['maptype']} (ms)"

        fig = plt.figure(figsize=(2.5, 5))
        cbar_ax = fig.add_axes([0.25, 0.05, 0.2, 0.9])
        norm = plt.Normalize(vmin=lo, vmax=hi)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(label, fontsize=11)
        cbar_ax.tick_params(labelsize=10)
        fig.canvas.manager.set_window_title(f'{label} Colorbar')
        plt.show(block=False)

    def on_left_volume_changed(self, index):
        self.left_asset_path = self.left_dropdown.itemData(index)
        self._show_colorbar_figure(self.left_asset_path)
        self.update_plots()

    def on_right_volume_changed(self, index):
        self.right_asset_path = self.right_dropdown.itemData(index)
        self._show_colorbar_figure(self.right_asset_path)
        self.update_plots()

    def on_slice_changed(self, value):
        self.slice_label.setText(f"Slice: {value + 1} / {self.num_slices}")
        self.update_plots()

    def on_opacity_changed(self, value):
        opacity = self.opacity_levels[value]
        self.opacity_label.setText(f"Overlay Opacity: {int(opacity * 100)}%")
        self.refresh_overlay_panel()

    def get_panel_image(self, path, hne, img_number):
        """Resolves an asset path to (image_array, title, cmap, vmin, vmax, colorbar_label, missing_message).

        image_array is None when the asset can't be loaded for this slice;
        missing_message then holds the placeholder text to display instead.
        colorbar_label is non-None only for relaxation maps (T1/T2).
        """
        if path == 'HNE':
            return hne, f"H&E Histology (Img: {img_number})", None, None, None, None, None
        elif path == 'BF':
            bf_image = self.load_blockface_slice(img_number)
            if bf_image is not None:
                return bf_image, f"Blockface RGB (Img: {img_number})", None, None, None, None, None
            return None, "Blockface Missing", None, None, None, None, f"Missing BF: {img_number}"
        else:
            vol_arr = self.load_volume(path)
            slice_num = self.get_bf_slice_index(img_number)

            if slice_num < vol_arr.shape[2]:
                MR_Slice = vol_arr[:, :, slice_num].T
            else:
                MR_Slice = np.zeros((hne.shape[0] // 4, hne.shape[1] // 4))

            vol_upsampled = np.repeat(np.repeat(MR_Slice, 4, axis=0), 4, axis=1)
            MR_Slice_us = vol_upsampled[:hne.shape[0], :hne.shape[1]]

            title = f"MR: {os.path.basename(path)} (Slice: {slice_num})"
            style = get_mr_style(os.path.basename(path))
            if style['kind'] == 'relaxation':
                lo, hi = style['loLev'], style['upLev']
                _, lut = relaxation(style['maptype'], np.array([lo, hi]), lo, hi)
                cmap = ListedColormap(lut)
                return MR_Slice_us, title, cmap, lo, hi, f"{style['maptype']} (ms)", None
            else:
                return MR_Slice_us, title, style['cmap'], None, None, None, None

    def draw_panel_data(self, ax, path, hne, img_number):
        image, title, cmap, vmin, vmax, colorbar_label, missing_message = self.get_panel_image(path, hne, img_number)
        if image is not None:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.text(0.5, 0.5, missing_message, color="orange", ha="center", va="center")
        ax.set_title(title)

    def draw_overlay_panel(self, ax, hne, img_number):
        """Draws the left asset on the bottom and the right asset on top,
        blended via the opacity slider, so alignment can be checked by eye."""
        left_image = left_cmap = left_vmin = left_vmax = left_cblabel = None
        if self.left_asset_path is not None:
            left_image, _, left_cmap, left_vmin, left_vmax, left_cblabel, _ = self.get_panel_image(self.left_asset_path, hne, img_number)

        right_image = right_cmap = right_vmin = right_vmax = right_cblabel = None
        if self.right_asset_path is not None:
            right_image, _, right_cmap, right_vmin, right_vmax, right_cblabel, _ = self.get_panel_image(self.right_asset_path, hne, img_number)

        if left_image is None and right_image is None:
            ax.text(0.5, 0.5, "No Assets Selected", color="gray", ha="center", va="center")
            ax.set_title("Overlay: Empty")
            return

        opacity = self.opacity_levels[self.opacity_slider.value()]
        if left_image is not None:
            ax.imshow(left_image, cmap=left_cmap, vmin=left_vmin, vmax=left_vmax)
        if right_image is not None:
            ax.imshow(right_image, cmap=right_cmap, vmin=right_vmin, vmax=right_vmax, alpha=opacity)
        ax.set_title(f"Overlay (opacity: {opacity:.0%})")

    def capture_axis_view(self, ax):
        """Returns (image_shapes, xlim, ylim) for an axis currently showing
        image(s), or None if it's empty (e.g. a placeholder-text axis)."""
        images = ax.get_images()
        if not images:
            return None
        return (tuple(im.get_array().shape for im in images), ax.get_xlim(), ax.get_ylim())

    def restore_axis_view(self, ax, prior):
        """Re-applies a view captured by capture_axis_view, but only if the
        freshly redrawn axis holds image(s) of the same shape(s) as before --
        otherwise a placeholder axis's default (0, 1) range, or a differently
        shaped image from switching assets, would be applied as a bogus zoom."""
        if prior is None:
            return
        prior_shapes, prior_xlim, prior_ylim = prior
        images = ax.get_images()
        shapes = tuple(im.get_array().shape for im in images)
        if images and shapes == prior_shapes:
            ax.set_xlim(prior_xlim)
            ax.set_ylim(prior_ylim)

    def draw_tile_grid(self, axes, hne):
        origin_list = self.tiling_tool(hne, self.tilesize)
        if origin_list.ndim != 2 or len(origin_list) == 0:
            return

        color_palette = ['#ff8b17', '#ffe417', '#b2ff17', '#36ff17', '#17fbff', '#17a6ff']
        for q in range(len(origin_list)):
            row, col = origin_list[q, 0], origin_list[q, 1]
            y = row // self.tilesize
            x = col // self.tilesize
            color = color_palette[(y + x) % len(color_palette)]

            for ax in axes:
                rect = patches.Rectangle((col, row), self.tilesize, self.tilesize,
                                         linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

    def update_plots(self):
        idx = self.slider.value()

        # Preserve any zoom/pan set via the navigation toolbar across redraws,
        # since ax.clear() resets axis limits to the image's full extent.
        prior_views = [self.capture_axis_view(ax) for ax in self.axes]

        for ax in self.axes:
            ax.clear()

        hne_ds_im = self.reg_HnE_arr[:, :, idx, :]
        img_number = re.search(r'\d+', self.hne_filenames[idx]).group()

        overlay_ax, left_ax, right_ax = self.axes

        self.draw_overlay_panel(overlay_ax, hne_ds_im, img_number)

        if self.left_asset_path is not None:
            self.draw_panel_data(left_ax, self.left_asset_path, hne_ds_im, img_number)
        else:
            left_ax.text(0.5, 0.5, "No Asset Selected", color="gray", ha="center", va="center")
            left_ax.set_title("Left View: Empty")

        if self.right_asset_path is not None:
            self.draw_panel_data(right_ax, self.right_asset_path, hne_ds_im, img_number)
        else:
            right_ax.text(0.5, 0.5, "No Asset Selected", color="gray", ha="center", va="center")
            right_ax.set_title("Right View: Empty")

        if self.grid_checkbox.isChecked():
            self.draw_tile_grid(self.axes, hne_ds_im)

        for ax, prior in zip(self.axes, prior_views):
            self.restore_axis_view(ax, prior)

        for ax in self.axes:
            ax.axis('off')
        self.figure.tight_layout(pad=1.5)
        self.figure.subplots_adjust(wspace=0.25)
        self.canvas.draw()

    def refresh_overlay_panel(self):
        """Redraws only the overlay panel. Used on opacity changes, since the
        left/right panels and the tile grid don't depend on opacity -- routing
        every slider tick through the full update_plots redraw is what made
        dragging the opacity slider feel laggy."""
        idx = self.slider.value()
        hne_ds_im = self.reg_HnE_arr[:, :, idx, :]
        img_number = re.search(r'\d+', self.hne_filenames[idx]).group()

        ax = self.axes[0]
        prior = self.capture_axis_view(ax)

        ax.clear()
        self.draw_overlay_panel(ax, hne_ds_im, img_number)
        if self.grid_checkbox.isChecked():
            self.draw_tile_grid([ax], hne_ds_im)

        self.restore_axis_view(ax, prior)
        ax.axis('off')
        self.canvas.draw()

RabbitFolder='/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
RabbitID="R24-101"
Block = 11

if __name__ == '__main__':
    blockId = "Block" + f"{Block:02d}"

    # POINT THIS TO THE FOLDER CONTAINING INTERM EDIATE NII.GZ VOLUMES
    # ROOT folder that branches into InVivo_MR/, ExVivo_MR/, etc.
    rabbit_mr_root = os.path.join(RabbitFolder, RabbitID)
    hne_base = os.path.join(rabbit_mr_root, 'HnE', blockId)

    rabbase = os.path.split(os.path.split(hne_base)[0])[0]
    bf_cropped = os.path.join(rabbase, 'BlockFace_RGB', blockId, 'CroppedImages')

    app = QApplication(sys.argv)
    viewer = TillingGUI(hne_base, rabbit_mr_root, bf_cropped, blockId)
    viewer.show()
    sys.exit(app.exec_())
