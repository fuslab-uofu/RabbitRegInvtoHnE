import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


#Load in two images (preregistered)
impath="/Users/jbonaventura/Downloads/R24-082/Blockface_RGB/block06/CroppedImages/IMG_0043_scatter.tiff"
img = Image.open(impath)
ImArray= np.array(img)

maskpath='/Users/jbonaventura/Downloads/R24-082/HnE/block06/SegMask/IMG_0043_Reg.png'
mask = Image.open(maskpath)
MaskArray= np.array(mask)

#plt.imshow(ImArray)
plt.imshow(MaskArray)
plt.show()

#We can make this better later, but for now->
v1=153
v2=212
ImArray[:,:,:]=np.where(MaskArray[:,:,:]==v1,v1,ImArray[:,:,:])
ImArray[:,:,:]=np.where(MaskArray[:,:,:]==v2,v2,ImArray[:,:,:])
ImArray[:,:,:]=np.where(MaskArray==255,255,ImArray[:,:,:])
#ImArray[:,:,:]=np.where(MaskArray[:,:,:]==51,51,ImArray[:,:,:])
#ImArray[:,:,:]=np.where(MaskArray[:,:,:]==102,102,ImArray[:,:,:])

plt.imshow(ImArray)
plt.show()

#Save-
save_dir= os.path.dirname(maskpath)
filename= os.path.splitext(os.path.basename(maskpath))[0]
new_file_path = os.path.join(save_dir, filename + "burn.tiff")

CImage = Image.fromarray(ImArray.astype(np.uint8))
CImage.save(new_file_path, 'tiff')