import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VolumeViewer(QMainWindow):
    def __init__(self, data1, data2, title='3D Volume Slicer', seed_callback=None):
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.depth = data1.shape[2]
        self.current_slice = self.depth // 2
        self.opacity = 0
        self.seed_callback = seed_callback
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

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        # imshow of data1[:, :, sl] maps axis-0 to rows (ydata) and axis-1 to columns (xdata)
        x = int(round(event.ydata))
        y = int(round(event.xdata))
        z = self.current_slice
        self.seed_callback(x, y, z)

    def slider_value_changed2(self, value2):
        self.opacity = value2 / 100
        self.update_plot()
        self.slider_label2.setText(f"Opacity: {self.opacity}")

    def slider_value_changed(self, value):
        self.current_slice = value
        self.update_plot()
        self.slider_label.setText(f"Slice: {self.current_slice}/{self.depth - 1}")

    def update_plot(self):
        self.ax.clear()
        self.ax.imshow(self.data1[:, :, self.current_slice], vmin=0, vmax=255, cmap='viridis', origin='lower')
        self.ax.imshow(self.data2[:, :, self.current_slice], vmin=0, vmax=255, alpha=self.opacity, cmap='viridis', origin='lower')
        self.ax.set_title(f"Z-Slice {self.current_slice}")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.canvas.draw()
