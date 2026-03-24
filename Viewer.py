import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VolumeViewer(QMainWindow):
    def __init__(self, data1, data2, title='3D Volume Slicer', seed_callback=None,
                 undo_callback=None, cut_callback=None, expand_to_max_callback=None,
                 save_callback=None, load_callback=None):
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.slice_axis = int(np.argmin(data1.shape))
        self.display_axes = [i for i in range(data1.ndim) if i != self.slice_axis]
        self.depth = data1.shape[self.slice_axis]
        self.current_slice = self.depth // 2
        self.opacity = 0
        self.cut_radius = 1
        self.expand_to_max = False
        self.seed_callback = seed_callback
        self.undo_callback = undo_callback
        self.cut_callback = cut_callback
        self.expand_to_max_callback = expand_to_max_callback
        self.save_callback = save_callback
        self.load_callback = load_callback
        self._cut_mode = False
        self.setMouseTracking(True)
        self.setWindowTitle(title)
        self.initUI()
        self.update_plot()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

        slider_layout = QHBoxLayout()
        self.slider_label = QLabel(f"Slice: {self.current_slice}/{self.depth - 1}")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.depth - 1)
        self.slider.setValue(self.current_slice)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slider_value_changed)
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.slider)
        main_layout.addLayout(slider_layout)

        slider_layout2 = QHBoxLayout()
        self.slider_label2 = QLabel(f"Opacity: {self.opacity}")
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(100)
        self.slider2.setValue(50)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(1)
        self.slider2.valueChanged.connect(self.slider_value_changed2)
        slider_layout2.addWidget(self.slider_label2)
        slider_layout2.addWidget(self.slider2)
        main_layout.addLayout(slider_layout2)

        if self.seed_callback is not None:
            self.canvas.mpl_connect('button_press_event', self._on_click)

        if self.load_callback is not None:
            load_btn = QPushButton('Load Volume')
            load_btn.clicked.connect(self.load_callback)
            main_layout.addWidget(load_btn)

        if self.save_callback is not None:
            save_btn = QPushButton('Save Segmentation')
            save_btn.clicked.connect(self.save_callback)
            main_layout.addWidget(save_btn)

        if self.expand_to_max_callback is not None:
            self._expand_btn = QPushButton('Expand to Max: OFF')
            self._expand_btn.setCheckable(True)
            self._expand_btn.setStyleSheet('QPushButton:checked { background-color: #2980b9; color: white; }')
            self._expand_btn.toggled.connect(self._toggle_expand_to_max)
            main_layout.addWidget(self._expand_btn)

        if self.undo_callback is not None:
            undo_btn = QPushButton('Delete Last Point')
            undo_btn.clicked.connect(self.undo_callback)
            main_layout.addWidget(undo_btn)

        if self.cut_callback is not None:
            self._cut_btn = QPushButton('Cut Mode: OFF')
            self._cut_btn.setCheckable(True)
            self._cut_btn.setStyleSheet('QPushButton:checked { background-color: #c0392b; color: white; }')
            self._cut_btn.toggled.connect(self._toggle_cut_mode)
            main_layout.addWidget(self._cut_btn)

            cut_radius_layout = QHBoxLayout()
            self._cut_radius_label = QLabel(f'Cut Radius: {self.cut_radius}')
            self._cut_radius_slider = QSlider(Qt.Horizontal)
            self._cut_radius_slider.setMinimum(0)
            self._cut_radius_slider.setMaximum(10)
            self._cut_radius_slider.setValue(self.cut_radius)
            self._cut_radius_slider.setTickPosition(QSlider.TicksBelow)
            self._cut_radius_slider.setTickInterval(1)
            self._cut_radius_slider.valueChanged.connect(self._cut_radius_changed)
            cut_radius_layout.addWidget(self._cut_radius_label)
            cut_radius_layout.addWidget(self._cut_radius_slider)
            main_layout.addLayout(cut_radius_layout)

    def _toggle_expand_to_max(self, checked):
        self.expand_to_max = checked
        self._expand_btn.setText('Expand to Max: ON' if checked else 'Expand to Max: OFF')
        self.expand_to_max_callback()

    def _toggle_cut_mode(self, checked):
        self._cut_mode = checked
        self._cut_btn.setText('Cut Mode: ON' if checked else 'Cut Mode: OFF')

    def _cut_radius_changed(self, value):
        self.cut_radius = value
        self._cut_radius_label.setText(f'Cut Radius: {value}')

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        # imshow maps axis-0 of the 2D slice to rows (ydata) and axis-1 to columns (xdata)
        coords = [0, 0, 0]
        coords[self.display_axes[0]] = int(round(event.ydata))
        coords[self.display_axes[1]] = int(round(event.xdata))
        coords[self.slice_axis] = self.current_slice
        if self._cut_mode and self.cut_callback is not None:
            self.cut_callback(*coords)
        elif self.seed_callback is not None:
            self.seed_callback(*coords)

    def slider_value_changed2(self, value2):
        self.opacity = value2 / 100
        self.update_plot()
        self.slider_label2.setText(f"Opacity: {self.opacity}")

    def slider_value_changed(self, value):
        self.current_slice = value
        self.update_plot()
        self.slider_label.setText(f"Slice: {self.current_slice}/{self.depth - 1}")

    def reset(self, data1, data2):
        """Swap in a new volume and reset all slice/overlay state."""
        self.data1 = data1
        self.data2 = data2
        self.slice_axis = int(np.argmin(data1.shape))
        self.display_axes = [i for i in range(data1.ndim) if i != self.slice_axis]
        self.depth = data1.shape[self.slice_axis]
        self.current_slice = self.depth // 2
        self.slider.setMaximum(self.depth - 1)
        self.slider.setValue(self.current_slice)
        self.slider_label.setText(f'Slice: {self.current_slice}/{self.depth - 1}')
        if hasattr(self, '_expand_btn') and self._expand_btn.isChecked():
            self._expand_btn.setChecked(False)
        self.update_plot()

    def update_plot(self):
        sl = [slice(None)] * self.data1.ndim
        sl[self.slice_axis] = self.current_slice
        self.ax.clear()
        self.ax.imshow(self.data1[tuple(sl)], vmin=0, vmax=255, cmap='viridis', origin='lower')
        self.ax.imshow(self.data2[tuple(sl)], vmin=0, vmax=255, alpha=self.opacity, cmap='viridis', origin='lower')
        self.ax.set_title(f"Axis-{self.slice_axis} Slice {self.current_slice}")
        self.ax.set_xlabel(f"Axis {self.display_axes[1]}")
        self.ax.set_ylabel(f"Axis {self.display_axes[0]}")
        self.canvas.draw()
