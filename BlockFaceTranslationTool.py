import os
import re
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QSpinBox, QLabel, QGroupBox, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QPoint
import PyQt5.sip
sys.modules.setdefault('sip', PyQt5.sip)  # landmark .npy files pickle QPoint/QPointF objects,
                                           # whose unpickler does a bare `import sip`, not `PyQt5.sip`
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# Default folder the file pickers open into -- just a convenience starting
# point, any rabbit/block's CroppedImages folder can be browsed to instead.
DEFAULT_BROWSE_DIR = ('/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/'
                      'RabbitData/R24-104/BlockFace_RGB/Block06/CroppedImages')

# gui_he2blk.py (the tool that produces Registered/*.png + Landmarks/*.npy)
# saves landmark points in its own downscaled display coordinates
# (image dimensions // sf1234), not full-resolution pixel space -- confirmed
# from its `sf1234 = 3` constant and the `* sf1234` rescale it applies before
# actually using the points for a warp. Must match that constant exactly.
DISPLAY_SCALE = 3


def translate_image(img, dx, dy):
    """Shifts img by (dx, dy) pixels, padding revealed edges with black (0)
    and keeping the same array shape. Black matches this pipeline's existing
    background convention in BlockFace CroppedImages TIFFs (see
    BlockFaceToNifti in IntermediateRegProcessingPipeline.py), and content
    pushed past the opposite edge is simply cropped off to preserve size."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def parse_rabbit_block(path):
    """Pulls the RabbitID (e.g. R24-104) and Block label (e.g. Block06) out of
    a path under .../RabbitData/<RabbitID>/.../<BlockXX>/..., so translated
    output lands in its own per-rabbit/block subfolder instead of colliding
    with another rabbit's identically-named IMG_XXXX_scatter.tiff file."""
    parts = Path(path).parts
    rabbit_id = next((p for p in parts if re.fullmatch(r'R\d+-\d+', p)), "UnknownRabbit")
    block = next((p for p in parts if re.fullmatch(r'Block\d+', p)), "UnknownBlock")
    return rabbit_id, block


def _find_by_img_number(folder, suffix, img_int):
    """Matches on the numeric value of each filename's digits, not substring
    containment -- mirrors gui_he2blk.py's load_all_from_hne, which avoids a
    slice like 12 false-matching a file that happens to contain '112'."""
    if not folder.exists():
        return None
    candidates = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() == suffix and not f.name.startswith('._')
        and any(int(n) == img_int for n in re.findall(r'\d+', f.name))
    )
    return candidates[0] if candidates else None


def find_hne_paths(moving_path):
    """Given a BlockFace CroppedImages TIFF path, locates the matching
    Registered PNG and Landmarks .npy for the same slice (either, both, or
    neither may exist). Returns (registered_path, landmarks_path), each None
    if not found."""
    moving_path = Path(moving_path)
    parts = moving_path.parts
    try:
        block_idx = parts.index('BlockFace_RGB')
    except ValueError:
        return None, None

    rabbase = Path(*parts[:block_idx])
    block_name = parts[block_idx + 1]
    hne_block_dir = rabbase / 'HnE' / block_name

    img_int = int(re.search(r'\d+', moving_path.stem).group())
    reg_path = _find_by_img_number(hne_block_dir / 'Registered', '.png', img_int)
    lm_path = _find_by_img_number(hne_block_dir / 'Landmarks', '.npy', img_int)
    return reg_path, lm_path


class BlockFaceTranslationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlockFace Translation Tool")
        self.setGeometry(100, 100, 1000, 900)

        self.reference_path = None
        self.reference_arr = None
        self.moving_path = None
        self.moving_arr = None
        self.shifted_arr = None
        self.registered_path = None
        self.registered_arr = None
        self.landmarks_path = None
        self.landmarks = None

        self.init_ui()
        self.update_plot()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.figure = plt.figure(figsize=(9, 8))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        controls_box = QGroupBox("Alignment Controls")
        controls_layout = QVBoxLayout()

        # --- File pickers ---
        file_layout = QHBoxLayout()
        self.load_ref_btn = QPushButton("Load Reference Image...")
        self.load_ref_btn.clicked.connect(self.load_reference)
        self.ref_label = QLabel("Reference: (none)")
        self.load_moving_btn = QPushButton("Load Image to Translate...")
        self.load_moving_btn.clicked.connect(self.load_moving)
        self.moving_label = QLabel("Moving: (none)")
        file_layout.addWidget(self.load_ref_btn)
        file_layout.addWidget(self.ref_label, stretch=1)
        file_layout.addWidget(self.load_moving_btn)
        file_layout.addWidget(self.moving_label, stretch=1)
        controls_layout.addLayout(file_layout)

        self.hne_status_label = QLabel("Registered H&E / landmarks: (load a moving image first)")
        controls_layout.addWidget(self.hne_status_label)

        # --- X / Y translation sliders, each paired with a spinbox for exact entry ---
        self.x_slider, self.x_spin, x_row = self._make_translation_row("X shift (px):", self.on_shift_changed)
        self.y_slider, self.y_spin, y_row = self._make_translation_row("Y shift (px):", self.on_shift_changed)
        controls_layout.addLayout(x_row)
        controls_layout.addLayout(y_row)

        # --- Opacity slider ---
        opacity_layout = QHBoxLayout()
        self.opacity_label = QLabel("Overlay Opacity: 50%")
        self.opacity_label.setFixedWidth(150)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addWidget(self.opacity_slider, stretch=1)
        controls_layout.addLayout(opacity_layout)

        # --- Save + status ---
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Translated Image")
        self.save_btn.clicked.connect(self.save_translated)
        self.status_label = QLabel("")
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.status_label, stretch=1)
        controls_layout.addLayout(save_layout)

        controls_box.setLayout(controls_layout)
        main_layout.addWidget(controls_box)

    def _make_translation_row(self, label_text, on_change):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-500)
        slider.setMaximum(500)
        slider.setValue(0)

        spin = QSpinBox()
        spin.setMinimum(-500)
        spin.setMaximum(500)
        spin.setValue(0)

        # Slider and spinbox stay in sync with each other; either one can drive
        # the shift, so blockSignals prevents the mutual updates from looping.
        def slider_to_spin(value):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
            on_change()

        def spin_to_slider(value):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            on_change()

        slider.valueChanged.connect(slider_to_spin)
        spin.valueChanged.connect(spin_to_slider)

        row.addWidget(slider, stretch=1)
        row.addWidget(spin)
        return slider, spin, row

    def _set_shift_range(self, width, height):
        # Wide enough to push the moving image fully out of frame in either
        # direction, since a couple of the flipped-and-rotated slices could
        # plausibly need a shift close to the image's own dimensions.
        self.x_slider.setMinimum(-width)
        self.x_slider.setMaximum(width)
        self.x_spin.setMinimum(-width)
        self.x_spin.setMaximum(width)
        self.y_slider.setMinimum(-height)
        self.y_slider.setMaximum(height)
        self.y_spin.setMinimum(-height)
        self.y_spin.setMaximum(height)

    def load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", DEFAULT_BROWSE_DIR, "TIFF Images (*.tiff *.tif)")
        if not path:
            return
        self.reference_path = path
        self.reference_arr = np.array(Image.open(path))
        self.ref_label.setText(f"Reference: {os.path.basename(path)}")
        self.update_plot()

    def load_moving(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image to Translate", DEFAULT_BROWSE_DIR, "TIFF Images (*.tiff *.tif)")
        if not path:
            return
        self.moving_path = path
        self.moving_arr = np.array(Image.open(path))
        self.moving_label.setText(f"Moving: {os.path.basename(path)}")

        # Reset any shift left over from a previously loaded image, and size
        # the sliders to this image's own dimensions.
        h, w = self.moving_arr.shape[:2]
        self._set_shift_range(w, h)
        self.x_slider.setValue(0)
        self.y_slider.setValue(0)

        self._load_hne_pair(path)

        self.update_plot()

    def _load_hne_pair(self, moving_path):
        """Looks for a Registered H&E PNG + Landmarks .npy matching this
        BlockFace slice, so both can be shifted in step with it on save."""
        self.registered_path, self.landmarks_path = find_hne_paths(moving_path)

        self.registered_arr = (np.array(Image.open(self.registered_path))
                               if self.registered_path else None)
        self.landmarks = (np.load(str(self.landmarks_path), allow_pickle=True)
                          if self.landmarks_path else None)

        found = []
        found.append(f"Registered: {self.registered_path.name}" if self.registered_path
                    else "Registered: not found")
        found.append(f"Landmarks: {self.landmarks_path.name}" if self.landmarks_path
                    else "Landmarks: not found")
        self.hne_status_label.setText(" | ".join(found))

    def on_shift_changed(self):
        self.update_plot()

    def on_opacity_changed(self, value):
        self.opacity_label.setText(f"Overlay Opacity: {value}%")
        self.update_plot()

    def update_plot(self):
        self.ax.clear()

        if self.reference_arr is not None:
            if (self.moving_arr is not None
                    and self.reference_arr.shape[:2] != self.moving_arr.shape[:2]):
                self.status_label.setText(
                    "WARNING: reference and moving image sizes differ -- overlay may look misleading")
            self.ax.imshow(self.reference_arr)

        if self.moving_arr is not None:
            dx = self.x_slider.value()
            dy = self.y_slider.value()
            self.shifted_arr = translate_image(self.moving_arr, dx, dy)
            opacity = self.opacity_slider.value() / 100
            self.ax.imshow(self.shifted_arr, alpha=opacity)

        if self.reference_arr is None and self.moving_arr is None:
            self.ax.text(0.5, 0.5, "Load a reference and/or moving image to begin",
                         color="gray", ha="center", va="center")

        self.ax.axis('off')
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_translated(self):
        if self.moving_path is None or self.shifted_arr is None:
            self.status_label.setText("Load an image to translate first.")
            return

        # Saved to the Desktop rather than back onto ceph -- ceph is a pain to
        # work with for repeated small writes, and this also keeps a second
        # copy out of CroppedImages itself, which BlockFaceToNifti() globs
        # directly (a second file there would double-count this slice).
        moving_name = os.path.basename(self.moving_path)
        rabbit_id, block = parse_rabbit_block(self.moving_path)
        out_dir = os.path.join(os.path.expanduser("~/Desktop"),
                               "BlockFaceTranslationTool_Output", rabbit_id, block)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, moving_name)

        Image.fromarray(self.shifted_arr).save(out_path)
        dx, dy = self.x_slider.value(), self.y_slider.value()
        saved = [moving_name]

        # Registered H&E PNG shares the BlockFace TIFF's exact pixel grid
        # (verified: both 2150x2800 for R24-104/Block06), so the identical
        # shift keeps them aligned with each other.
        if self.registered_arr is not None:
            if self.registered_arr.shape[:2] == self.moving_arr.shape[:2]:
                shifted_reg = translate_image(self.registered_arr, dx, dy)
                Image.fromarray(shifted_reg).save(os.path.join(out_dir, self.registered_path.name))
                saved.append(self.registered_path.name)
            else:
                self.status_label.setText(
                    "WARNING: Registered PNG size differs from moving image -- skipped, saved BlockFace only")

        # Landmarks are stored in gui_he2blk.py's downscaled display
        # coordinates, so the pixel shift is divided by DISPLAY_SCALE.
        # landmarks[0] is BlockFace-side (shifts with the image);
        # landmarks[1] is H&E-side (unchanged -- the H&E slide didn't move).
        if self.landmarks is not None:
            shifted_bf_points = [
                QPoint(int(round(p.x() + dx / DISPLAY_SCALE)), int(round(p.y() + dy / DISPLAY_SCALE)))
                for p in self.landmarks[0]
            ]
            hne_points = list(self.landmarks[1])
            np.save(os.path.join(out_dir, self.landmarks_path.name), [shifted_bf_points, hne_points])
            saved.append(self.landmarks_path.name)

        print(f"Saved (dx={dx}, dy={dy}) -> {out_dir}: {', '.join(saved)}")
        self.status_label.setText(f"Saved (dx={dx}, dy={dy}) -> {out_dir}: {', '.join(saved)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    tool = BlockFaceTranslationTool()
    tool.show()
    sys.exit(app.exec_())
