from pathlib import Path
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def FlattenBack2D(Varray, Myrange):
    ChunkOI=Varray[:,Myrange[0]:Myrange[1]]
    OneD=np.mean(ChunkOI, axis=1)
    # plt.plot(OneD)
    # plt.show()
    # cutrange1= int(input("Select SubRegion start"))
    # cutrange2= int(input("Select SubRegion end"))
    # print(cutrange1, cutrange2)
#    Slope=OneD[cutrange1:cutrange2]
    Slope=OneD
    invSlope=1/Slope
    inVSlope=invSlope/np.max(invSlope)
    AppliedChunk=Varray
    inVSlope_reshaped=inVSlope[:,np.newaxis]
    FlatChunk=AppliedChunk*inVSlope_reshaped

    #insert FlatChunk back into full array=
    FlatGuy=np.zeros_like(Varray)
    FlatGuy[:, :]=FlatChunk

    return FlatGuy

def rgbtogrey(img_rgb):
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    img_gray = 0.3* r/b + 0.35 * g/b + 0 * b
    #Flatten-
    img_gray=FlattenBack2D(img_gray,(0,300))
    #Normalize-
    img_graythresh=np.where(img_gray > 0.25, img_gray, 0)
    return img_gray, img_graythresh


def rgbtogrey2(img_rgb):
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    img_gray =255- (0.3* r + 0.35 * g + 0.35 * b)
    img_gray=np.where(img_rgb[:,:,0]==0, 0, img_gray)
    return img_gray


directory_path = '/Users/jbonaventura/Downloads/R23-055/BlockFace/block06/CroppedImages'
save_dir= os.path.join(os.path.dirname(directory_path), "greyIms3")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# For non-recursive search of .txt files in the current directory
files = [f for f in sorted(os.listdir(directory_path)) if f.endswith('tiff')]

#Take a handful of these for testing
fileschunk=files[:5]
ImList=[]
RawImList=[]
for file in files:
    print(file)
    full_path = os.path.join(directory_path, file)
    img = Image.open(full_path)
    ImArray= np.array(img)
    RawImList.append(ImArray)
    print(ImArray.shape)
    ImGrey, ImGreyThresh = rgbtogrey(ImArray)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    # # Display the images in the respective subplots
    # ax1.imshow(ImArray)
    # ax2.imshow(ImGrey)
    # ax3.imshow(ImGreyThresh)
    # plt.show()

    #Stack Thresholded Images into an array-
    ImList.append(ImGreyThresh)

    #Save to new folder-
    # new_file_path = os.path.join(save_dir, file)
    # CImage = Image.fromarray(ImGreyThresh)
    # CImage.save(new_file_path, 'TIFF')

ImArray=np.array(ImList)
ImArray_bino=np.where(ImArray > 0, 1, 0).astype(np.float64)
#ImArray_next=TakeOutBack(ImArray,30,200,10)

ks=10
box_blur_kernel = np.ones((ks, ks), np.float32) / ks**2
ImArrayProcessed=np.zeros_like(ImArray)

for i in range(ImArray.shape[1]):
    Imup=ImArray_bino[:,i,:]
    box_blur = cv2.filter2D(src=Imup, ddepth=-1, kernel=box_blur_kernel)
    ImArrayProcessed[:,i,:] = box_blur


for k in range(ImArrayProcessed.shape[0]):
    Imup=ImArrayProcessed[k,:,:]
    Mask=np.where(Imup>0.65,1,0)
    #Okay now multiply that mask times the original rgb im and shit them to a visually apealing greyscale-
    RGBImup=RawImList[k]
    #MaskReshaped=Mask[:,:,np.newaxis]
    #Multiply times mask
    #Does this work
    #MaskedRGB=RGBImup*MaskReshaped
    MaskedGrey=rgbtogrey2(RGBImup)
    # plt.imshow(MaskedGrey)
    # plt.show()
#    print(MaskedGrey.dtype)
    # Save to folder-
    new_file_path = os.path.join(save_dir, str(k)+"tester.tiff")
    CImage = Image.fromarray(MaskedGrey.astype(np.uint8))
    CImage.save(new_file_path, 'TIFF')


# for i in range(20):
#     Imup=ImArray[i*2,:,:]
#     Imup2=ImArrayProcessed[i*2,:,:]
#     Imup3=np.where(Imup2>0.75,1,0)
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     # Display the images in the respective subplots
#     ax1.imshow(Imup)
#     ax2.imshow(Imup2)
#     ax3.imshow(Imup3)
#     plt.show()










