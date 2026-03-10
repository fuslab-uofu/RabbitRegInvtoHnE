import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyQt5.QtGui import QPixmap, QMouseEvent, QImage, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import affine_transform
from skimage.filters import frangi
import SimpleITK as sitk

def maximum_intensity_projection(data, axis=2, slicesize=4):
    # 3. Perform the MIP using numpy.max() along the specified axis
    # The result will be a 2D array
    if axis == 0:
        mipimage = np.zeros((data.shape[0] - slicesize + 1, data.shape[1], data.shape[2]))
        for i in range(mipimage.shape[axis]):
            datachunk = data[i:i + slicesize, :, :]
            mipimage[i, :, :] = np.max(datachunk, axis=axis)
    if axis == 1:
        mipimage = np.zeros((data.shape[0], data.shape[1]- slicesize + 1, data.shape[2]))
        for i in range(mipimage.shape[axis]):
            datachunk = data[:,i:i+slicesize,:]
            mipimage[:,i,:]=np.max(datachunk, axis=axis)
    if axis == 2:
        mipimage = np.zeros((data.shape[0], data.shape[1], data.shape[2]- slicesize + 1))
        for i in range(mipimage.shape[axis]):
            datachunk = data[:,:,i:i+slicesize]
            mipimage[:,:,i]=np.max(datachunk, axis=axis)

    return mipimage

# Example usage:
# Replace 'path/to/your/scan.nii.gz' with the actual path to your file.
# You can change the 'axis' parameter to project along different directions (0=sagittal, 1=coronal, 2=axial)



class VolumeViewer(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.depth = data.shape[2]
        self.mipwindow=1
        self.current_slice = self.depth // 2
        self.thresholding=0
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
        self.slider_label2 = QLabel(f"Thresholding: {self.thresholding}")
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(255)
        self.slider2.setValue(0)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(1)
        # Connect slider to update function
        self.slider2.valueChanged.connect(self.slider_value_changed2)
        slider_layout2.addWidget(self.slider_label2)
        slider_layout2.addWidget(self.slider2)
        main_layout.addLayout(slider_layout2)

        slider_layout3 = QHBoxLayout()
        self.slider_label3 = QLabel(f"MIP Window: {self.mipwindow}")
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setMinimum(0)
        self.slider3.setMaximum(40)
        self.slider3.setValue(0)
        self.slider3.setTickPosition(QSlider.TicksBelow)
        self.slider3.setTickInterval(1)
        # Connect slider to update function
        self.slider3.valueChanged.connect(self.slider_value_changed3)
        slider_layout3.addWidget(self.slider_label3)
        slider_layout3.addWidget(self.slider3)
        main_layout.addLayout(slider_layout3)

    def slider_value_changed3(self, value3):
        self.mipwindow = value3
        self.update_plot()
        self.slider_label3.setText(f"MIP Window: {self.mipwindow}")

    def slider_value_changed2(self, value2):
        self.thresholding = value2
        self.update_plot()
        self.slider_label2.setText(f"Thresholding: {self.thresholding}")

    def slider_value_changed(self, value):
        self.current_slice = value
        self.update_plot()
        self.slider_label.setText(f"Slice: {self.current_slice}/{self.depth - 1}")

    def update_plot(self):
        # Clear the previous plot and display the new slice
        #Update data with thresholding-
        self.datanew = np.where(self.data> self.thresholding, self.data, 0)
        self.datanew=maximum_intensity_projection(self.datanew, axis=2, slicesize=self.mipwindow)
        self.depth = self.datanew.shape[2]
        self.ax.clear()
        # Display the 2D slice using imshow
        self.ax.imshow(self.datanew[:,:,self.current_slice], vmin=0,vmax=255,cmap='gray', origin='lower')
        self.ax.set_title(f"Z-Slice {self.current_slice}")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        # Redraw the canvas to reflect changes
        self.canvas.draw()

def FlattenBack(Varray, Myrange):
    #Want to generalize this to any axis
    if Myrange == []:
        Myrange = (0, Varray.shape[2])
    ChunkOI=Varray[:,:,Myrange[0]:Myrange[1]]
    #Need to normalize by number of non-zero elements too
    NoZeroCount=[]
    for i in range(Myrange[1]-Myrange[0]):
        nonZ=np.count_nonzero(ChunkOI[:,:,i])
        NoZeroCount.append(nonZ)
    NonZeroVec=np.array(NoZeroCount)
    NonZeroVec=np.where(NonZeroVec==0,1,NonZeroVec)
    plt.plot(NonZeroVec)
    plt.show()
    OneD=np.mean(np.mean(ChunkOI, axis=0),axis=0)/NonZeroVec
    plt.plot(OneD)
    plt.show()
    cutrange1= int(input("Select SubRegion start"))
    cutrange2= int(input("Select SubRegion end"))
    print(cutrange1, cutrange2)
    Slope=OneD[cutrange1:cutrange2]
    Slope=Slope
    invSlope=1/Slope
    inVSlope=invSlope/np.max(invSlope)
    AppliedChunk=Varray[:,:,cutrange1:cutrange2]
    inVSlope_reshaped=inVSlope[np.newaxis,np.newaxis,:]
    FlatChunk=AppliedChunk*inVSlope_reshaped
    #Normalize back to 0-255
    FlatChunk=FlatChunk/np.max(FlatChunk)*255

    plt.imshow(FlatChunk[:,:,30])
    plt.show()

    #insert FlatChunk back into full array=
    FlatGuy=np.zeros_like(Varray)
    FlatGuy[:, :, cutrange1:cutrange2]=FlatChunk

    return FlatGuy


def main():
    # 1. Generate sample 3D data (e.g., a simple gradient volume)
    # Load in Nifti from path-
    nifti_path = '/Users/jbonaventura/Downloads/R23-055/InVivo_MR/48 3D_VIBE_0.5x0.5x1_cor_postContrast.nii'
    img = nib.load(nifti_path)
    affine_transform=img.affine
    nifti_array = img.get_fdata()
    nifti_array=nifti_array/np.max(nifti_array)*255

    #Thresholding-
    #nifti_thresh=50
    #nifti_post_thresh=np.where(nifti_array>nifti_thresh,nifti_array,0)

    # #Flat field correction-
    # flatguy= FlattenBack(nifti_post_thresh, [])
    #
    # #Then threshold again and save-
    # flatguy=np.where(flatguy>69,flatguy,0)

    #miparray=maximum_intensity_projection(nifti_post_thresh, axis=2, slicesize=20)


    img_sitk = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = otsu.Execute(img_sitk)
    mask_array = sitk.GetArrayFromImage(mask)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([20, 20, 20])
    corrected = corrector.Execute(img_sitk, mask)
    nifti_array = sitk.GetArrayFromImage(corrected)  # remember: (Z,Y,X) ordering
    nifti_array = nifti_array/np.max(nifti_array)*255
    nifti_array = np.transpose(nifti_array, (2, 1, 0))

    nifti_thresh=90
    nifti_post_thresh=np.where(nifti_array>nifti_thresh,nifti_array,0)




    # vesselness = frangi(nifti_array, sigmas=range(1, 8), black_ridges=False, gamma=4)
    # print(nifti_array.shape, vesselness.shape, np.max(vesselness), np.min(vesselness))
    # #Normalize-
    # vesselness=vesselness/np.max(vesselness)*100
    # vesselness=np.where(vesselness<5,0,vesselness)

   #  print(f"vesselness_norm min: {vesselness_norm.min():.4f}")
   #  print(f"vesselness_norm max: {vesselness_norm.max():.4f}")
   #  print(f"vesselness_norm mean: {vesselness_norm.mean():.6f}")
   #  print(f"vesselness_norm 99th percentile: {np.percentile(vesselness_norm, 99):.6f}")
   #
   #
   # # Step 1: Convert to SimpleITK (transpose back to Z,Y,X for SimpleITK)
   #  vessel_sitk = sitk.GetImageFromArray(
   #      np.transpose(vesselness_norm, (2, 1, 0)).astype(np.float32)
   #  )
   #
   #  # Step 2: Create edge potential image
   #  # Sigmoid maps high vesselness → near 0 (stops level set at vessel boundaries)
   #  # Alpha controls slope, Beta controls midpoint - tune these to your data
   #  sigmoid = sitk.SigmoidImageFilter()
   #  sigmoid.SetAlpha(-0.003)
   #  sigmoid.SetBeta(0.005)
   #  sigmoid.SetOutputMinimum(0.0)
   #  sigmoid.SetOutputMaximum(1.0)
   #  edge_potential = sigmoid.Execute(vessel_sitk)
   #  edge_array = sitk.GetArrayFromImage(edge_potential)
   #  print(f"edge potential min: {edge_array.min():.4f}, max: {edge_array.max():.4f}, mean: {edge_array.mean():.4f}")
   #  print(f"values below 0.1: {np.sum(edge_array < 0.1)}")
   #  print(f"values below 0.5: {np.sum(edge_array < 0.5)}")
   #
   #  # Step 3: Initialize level set from thresholded vesselness
   #  # This is your starting contour S - tune threshold to your data
   #  init_threshold = 0.012
   #  binary_init = (vesselness_norm > init_threshold).astype(np.float32)
   #  init_sitk = sitk.GetImageFromArray(
   #      np.transpose(binary_init, (2, 1, 0))
   #  )
   #
   #  print(f"init nonzero voxels: {np.count_nonzero(binary_init)}")
   #  # Visualize a slice
   #  slice_idx = binary_init.shape[2] // 2
   #  plt.imshow(binary_init[:, :, slice_idx], cmap='gray')
   #  plt.title("Initial contour")
   #  plt.show()
   #
   #  # Convert binary mask to signed distance function
   #  # (negative inside vessels, positive outside - this is S)
   #  distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
   #  distance_filter.SetInsideIsPositive(False)
   #  distance_filter.SetSquaredDistance(False)
   #  distance_filter.SetUseImageSpacing(True)
   #  init_sitk = sitk.Cast(init_sitk, sitk.sitkUInt32)
   #  initial_ls = distance_filter.Execute(init_sitk)
   #
   #  for n_iter in [50, 100, 200, 500, 1000]:
   #      geodesic = sitk.GeodesicActiveContourLevelSetImageFilter()
   #      geodesic.SetPropagationScaling(10.0)
   #      geodesic.SetCurvatureScaling(0.05)
   #      geodesic.SetAdvectionScaling(1.0)
   #      geodesic.SetNumberOfIterations(n_iter)
   #
   #      result_ls = geodesic.Execute(initial_ls, edge_potential)
   #      result_array = sitk.GetArrayFromImage(result_ls)
   #      refined = np.transpose((result_array <= 0).astype(np.float32), (2, 1, 0))
   #
   #      print(f"iter={n_iter}: nonzero voxels={np.count_nonzero(refined)}")
   #
   #      # Show same slice as your Frangi visualization
   #      plt.figure()
   #      plt.imshow(refined[:, :, slice_idx], cmap='gray')
   #      plt.title(f"Level set at {n_iter} iterations")
   #      plt.show()
   #
   #
   #  # Step 4: Run Geodesic Active Contour Level Set
   #  geodesic = sitk.GeodesicActiveContourLevelSetImageFilter()
   #  geodesic.SetPropagationScaling(10)  # expansion speed
   #  geodesic.SetCurvatureScaling(0.05)  # smoothness (higher = smoother)
   #  geodesic.SetAdvectionScaling(1.0)  # attraction to vessel boundaries
   #  geodesic.SetMaximumRMSError(0.01)  # convergence threshold
   #  geodesic.SetNumberOfIterations(1000)
   #
   #  result_ls = geodesic.Execute(initial_ls, edge_potential)
   #
   #  # Step 5: Extract binary segmentation (inside = negative values)
   #  result_array = sitk.GetArrayFromImage(result_ls)
   #  refined_segmentation = np.transpose(
   #      (result_array <= 0).astype(np.float32), (2, 1, 0)  # back to X,Y,Z
   #  )
   #
   #  print(np.max(refined_segmentation), np.min(refined_segmentation), refined_segmentation.shape)
   #
   #  # Save to NIfTI
   #  ni_img = nib.Nifti1Image(refined_segmentation, affine=affine_transform)
   #  nib.save(ni_img, '/Users/jbonaventura/Downloads/R24-103/In_Vivo_MR/LevelSetResult.nii')


    ni_img = nib.Nifti1Image(nifti_post_thresh, affine=affine_transform)
    # 4. Save the image to a file
    output_path = '/Users/jbonaventura/Downloads/R23-055/InVivo_MR/thresh.nii'  # Using .nii.gz for compressed output is common
    nib.save(ni_img, output_path)

 #   2. Run the PyQt5 application
    app = QApplication(sys.argv)
    viewer = VolumeViewer(nifti_post_thresh)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()