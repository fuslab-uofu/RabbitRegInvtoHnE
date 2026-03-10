import nibabel as nib
import h5py
import SimpleITK as sitk
import numpy as np
import sys
import cv2
from scipy.ndimage import label as ndimage_label
from PyQt5.QtGui import QPixmap, QMouseEvent, QImage, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os

def match_histograms(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    Zero voxels (background) in source are excluded from matching and remain zero.
    """
    src_values, src_lookup, src_counts = np.unique(source.reshape(-1), return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.reshape(-1), return_counts=True)

    src_zero_count=src_counts[0]
    tmpl_zero_count=tmpl_counts[0]
    src_counts[0]=0
    tmpl_counts[0]=0

    src_quantiles = np.cumsum(src_counts) / (source.size - src_zero_count)
    tmpl_quantiles = np.cumsum(tmpl_counts) / (template.size - tmpl_zero_count)
# Its all happening in this one line essentially->
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    matched = interp_a_values[src_lookup].reshape(source.shape)
    matched[source == 0] = 0
    return matched

def bin_intensities(vol, n_bins=8):
    """Quantize intensities into n_bins equal-width steps over [0, 255].
    Zero voxels (background) remain zero regardless of bin boundaries."""
    stepsize = 255 // n_bins
    binned = (np.floor(vol / stepsize) * stepsize).astype(np.float32)
    binned[vol == 0] = 0
    return binned

def importNifti(filepath):
    img = nib.load(filepath)
    affine_orig = img.affine
    nifti_array = img.get_fdata()
    #Normalize nifti array to standard image brightness-
    nifti_array=nifti_array/np.max(nifti_array)*255
    return(nifti_array,affine_orig)

# Applies a 2D box filter slice-by-slice to find tissue regions.
# Pixels above thresh are boosted to 1000 before filtering so dense tissue
# regions score much higher than sparse background after averaging.
def TakeOutBack(pic, thresh=70, maskthresh=150, kernsize=10):
    if np.max(pic) == 0:
        return np.zeros(pic.shape, dtype=np.uint8)

    pic = np.array(pic / np.max(pic) * 255).astype(np.uint8)
    thresh_pic = np.where(pic > thresh, 1000, pic).astype(np.float32)

    kernel = np.ones((kernsize, kernsize), np.float32) / (kernsize * kernsize)
    filtered = np.zeros(thresh_pic.shape, dtype=np.float32)
    for k in range(thresh_pic.shape[2]):
        filtered[:, :, k] = cv2.filter2D(thresh_pic[:, :, k], ddepth=-1, kernel=kernel)

    return (filtered > maskthresh).astype(np.uint8)

#Designed to work on a binary mask-
def AddBorder(CroppedMask, nupix):
    BorderedMask=CroppedMask.copy()
    for k in range(CroppedMask.shape[2]):
        sobel_x = cv2.Sobel(CroppedMask[:,:,k].astype(np.float64), cv2.CV_64F,1,0, ksize=3)
        sobel_y = cv2.Sobel(CroppedMask[:,:,k].astype(np.float64), cv2.CV_64F,0,1, ksize=3)
        derivtot= cv2.convertScaleAbs(sobel_x)+cv2.convertScaleAbs(sobel_y)
        NonZeroList=np.argwhere(derivtot !=0)
        if len(NonZeroList)>0:
            for ind in NonZeroList:
                BorderedMask[ind[0]-nupix:ind[0]+nupix, ind[1]-nupix:ind[1]+nupix, k]=1
    return (BorderedMask)

# New viewer with opacity slider-
class VolumeViewer(QMainWindow):
    def __init__(self, data1, data2):
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.depth = data1.shape[2]
        self.current_slice = self.depth // 2
        self.opacity=0
        self.setMouseTracking(True)
        self.initUI()
        self.update_plot()

    def initUI(self):
        self.setWindowTitle("3D Volume Slicer (PyQt5 + Matplotlib)")
        self.setGeometry(100, 100, 800, 600)
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        # Matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

        # Slider and label layout
        slider_layout = QHBoxLayout()
        self.slider_label = QLabel(f"Slice: {self.current_slice}/{self.depth - 1}")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.depth - 1)
        self.slider.setValue(self.current_slice)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        # Connect slider to update function
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
        # Connect slider to update function
        self.slider2.valueChanged.connect(self.slider_value_changed2)
        slider_layout2.addWidget(self.slider_label2)
        slider_layout2.addWidget(self.slider2)
        main_layout.addLayout(slider_layout2)

    def slider_value_changed2(self, value2):
        self.opacity = value2/100
        self.update_plot()
        self.slider_label2.setText(f"Opacity: {self.opacity}")

    def slider_value_changed(self, value):
        self.current_slice = value
        self.update_plot()
        self.slider_label.setText(f"Slice: {self.current_slice}/{self.depth - 1}")

    def update_plot(self):
        # Clear the previous plot and display the new slice
        #Update data with opacity-
        # self.datanew = np.where(self.data1> self.opacity, self.data1, 0)
        # self.depth = self.datanew.shape[2]
        self.ax.clear()
        # Display the 2D slice using imshow
        self.ax.imshow(self.data1[:,:,self.current_slice], vmin=0, vmax= 255, cmap='viridis', origin='lower')
        self.ax.imshow(self.data2[:, :, self.current_slice], vmin=0, vmax= 255, alpha=self.opacity, cmap='viridis', origin='lower')
        self.ax.set_title(f"Z-Slice {self.current_slice}")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        # Redraw the canvas to reflect changes
        self.canvas.draw()

def FlattenBack(Varray, Myrange):
    ChunkOI=Varray[Myrange[0]:Myrange[1],:,:]

    OneD=np.mean(np.mean(ChunkOI, axis=0),axis=1)
    plt.plot(OneD)
    plt.show()
    cutrange1= int(input("Select SubRegion start"))
    cutrange2= int(input("Select SubRegion end"))
    print(cutrange1, cutrange2)
    Slope=OneD[cutrange1:cutrange2]
    Slope=Slope
    invSlope=1/Slope
    inVSlope=invSlope/np.max(invSlope)
    AppliedChunk=Varray[:,cutrange1:cutrange2,:]
    inVSlope_reshaped=inVSlope[np.newaxis,:,np.newaxis]
    FlatChunk=AppliedChunk*inVSlope_reshaped

    #insert FlatChunk back into full array=
    FlatGuy=np.zeros_like(Varray)
    FlatGuy[:, cutrange1:cutrange2, :]=FlatChunk

    return FlatGuy



def makesegfromvol(vol_array, thresh=70, maskthresh=160, kernsize=10):
    mask = TakeOutBack(vol_array, thresh, maskthresh, kernsize)

    # Keep only the largest connected component — removes edge artifacts and noise
    labeled, n_components = ndimage_label(mask)
    if n_components == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # ignore background label
    largest_label = np.argmax(component_sizes)

    return (labeled == largest_label).astype(np.uint8)


def Segnifti(filepath):
    #loading in unmasked exv-
    ExV_array, ExV_affine = importNifti(filepath)
    FinalMask = makesegfromvol(ExV_array)
    #Mask ExV-
    ExVMasked= ExV_array*FinalMask

 #   2. Run the PyQt5 application
    app = QApplication(sys.argv)
    #For masking-
    viewer = VolumeViewer(ExV_array, ExVMasked)
    viewer.show()

    # Save the final segmentation
    ni_img = nib.Nifti1Image(FinalMask.astype(np.uint8), affine=ExV_affine)
    ExV_dir= os.path.dirname(filepath)
    output_path = os.path.join(ExV_dir, 'PyMask.nii.gz')
    nib.save(ni_img, output_path)
    sys.exit(app.exec_())
    return()

def DimsDivFour(volume):
    dims=volume.shape
    for i in range(len(dims)):
        if dims[i] % 4 != 0:
            for j in range(4-dims[i]%4):
                # Can we add a row to each one of these-
                if i==0:
                    extrabit = np.zeros_like(volume[0, :, :])
                    extrabit3d = extrabit[np.newaxis, :, :]
                if i==1:
                    extrabit = np.zeros_like(volume[:, 0, :])
                    extrabit3d = extrabit[:, np.newaxis, :]
                if i==2:
                    extrabit = np.zeros_like(volume[:, :, 0])
                    extrabit3d = extrabit[:, :, np.newaxis]
                volume = np.append(volume, extrabit3d, axis=i)
    return(volume)


def SegandHistoMatch(Seg_Path, ExV_path, InV_path):
    #Import Files-
    if Seg_Path != []:
        FinalMask = importNifti(Seg_Path)[0]
        FinalMask=FinalMask/np.max(FinalMask)

    ExV_array, ExV_affine = importNifti(ExV_path)
    InV_array, InV_affine = importNifti(InV_path)
    print(ExV_path, InV_path)
    print(ExV_array.shape,InV_array.shape)

    #Mask ExV-
    ExVMasked= ExV_array*FinalMask
    #Extend mask for InV->
    #FinalMask = AddBorder(FinalMask, 2)

    #InV_array = np.where(InV_array>20, InV_array, 0)
    # apply segmentation to nifti-
    masked_InV = InV_array*FinalMask
    masked_InV = DimsDivFour(masked_InV)

    #Parameters work as (Input to be Matched, Reference)
    matched_ExV = match_histograms(ExVMasked, masked_InV)
    matched_ExV = DimsDivFour(matched_ExV)

    #   2. Run the PyQt5 application
    app = QApplication(sys.argv)
    viewer = VolumeViewer(matched_ExV, masked_InV)
    viewer.show()

    # #Binarize these bad boys-
    # masked_InV=np.where(masked_InV>0, 2, 1)
    # matched_ExV=np.where(matched_ExV>0, 2, 1)

    #4. Save In Vivo image to a file
    ni_img = nib.Nifti1Image(masked_InV.astype(np.uint8), affine=InV_affine)
    InV_dir= os.path.dirname(InV_path)
    InVout_path = os.path.join(InV_dir, 'PyMaskedVolume_fixed.nii.gz')
    nib.save(ni_img, InVout_path)

    # Save Ex Vivo Histomatched to file
    ni_img = nib.Nifti1Image(matched_ExV.astype(np.uint8), affine=ExV_affine)
    ExV_dir= os.path.dirname(ExV_path)
    ExVout_path = os.path.join(ExV_dir, 'PyHistomatched_fixed.nii.gz')
    nib.save(ni_img, ExVout_path)
    sys.exit(app.exec_())
    return()

def main():
    #Animal Path-
    RabPath='/Users/jbonaventura/Downloads/R24-103'
    Block='Block06'
    blockname="ExVMR_"+Block+"Cropped.nii.gz"
    ExV_path = os.path.join(RabPath, 'ExVivo_MRBlocked',Block, blockname)
    seg_path = os.path.join(os.path.dirname(ExV_path), 'PyMask.nii.gz')

    blockfolder=Block+"Reg"
    blockname2="ExV"+Block+"Resampled.nii.gz"
    InV_path=os.path.join(RabPath,'ExVivo_MR', blockfolder, blockname2)

    #To Make Segment-
    Segnifti(ExV_path)

    #To apply segment and histomatch-
    #SegandHistoMatch(seg_path, ExV_path,InV_path)

 #   2. Run the PyQt5 application
 #    app = QApplication(sys.argv)
 #    #For masking-
 #    #viewer = VolumeViewer(ExV_array, ExVMasked)
 #    #For InV ExV Comp-
 #    viewer = VolumeViewer(matched_ExV, masked_InV)
 #    viewer.show()




if __name__ == "__main__":
    main()




