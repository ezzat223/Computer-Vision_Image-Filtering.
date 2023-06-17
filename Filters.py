import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import base64
from PIL import Image
import cv2
from io import StringIO 
from random import randint
from scipy import ndimage




def saveImg_unique(img,path, rgb=True, save_on_current = True):
    path_img = f"{path}{randint(1,100000)}.png"
    with open(path_img, 'wb') as f:
        
        cv2.imwrite(path_img, img)
    if save_on_current:
        with open('./static/img/input/current.png', 'wb') as f:
            
            cv2.imwrite('./static/img/input/current.png', img)
        
    return path_img


def read_rgb(gray=False,normalize = True):
    img = cv2.imread('./static/img/input/current.png') #0 is to read the as gray scale
    if gray:
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img=rgb2gray(img)
    img=cv2.resize(img,(640 , 426))
    # normalize the value of each pixel to be between 0 &1
    if normalize:
        img=img/225
    img = np.array(img)
    
    return img
'''
def read_img(encoded_data):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(encoded_data).split(',')[1])
    pimg = Image.open(sbuf)
    img = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
   

    return img
'''
def uniformNoise(a=0,b=0.2):
  img = read_rgb()
  
  x,y,_=img.shape
  #a=0
  #b=0.2
  uni_noise= np.zeros((x,y,3),dtype=np.float64)

  for color in (0,1,2):
      for i in range(x):
        for j in range(y):
          uni_noise[i][j][color]=np.random.uniform(a,b)
      noiseImg=uni_noise+img
  noiseImg=noiseImg*225

  path = saveImg_unique(noiseImg, "./static/img/output/")
  
  return path

def GaussianNoise(mean=0,sigma=0.223606797749979):
    img = read_rgb()
    x,y,_=img.shape
    # mean=0
    # var=0.05
    # sigma=np.sqrt(var)
    gauss_noise=np.random.normal(loc=mean,scale=sigma,size=(x,y,3))
    noiseImg=gauss_noise+img
    noiseImg=noiseImg*225
    path = saveImg_unique(noiseImg, "./static/img/output/")
  
    return path

def salt_pepperNoise(pepper=0.05):
    img = read_rgb()
    x,y,_=img.shape
    salt=1-pepper
    SP_noise= np.zeros((x,y,3),dtype=np.float64)
    for color in (0,1,2):
        for i in range(x):
          for j in range(y):
            rdn=np.random.random()
            if rdn< pepper:
              SP_noise[i][j][color]=0
            elif rdn > salt:
              SP_noise[i][j][color]=1
            else:
              SP_noise[i][j][color]=img[i][j][color]
        noiseImg=SP_noise
    noiseImg=noiseImg*225
    path = saveImg_unique(noiseImg, "./static/img/output/")
  
    return path

# #plot
# fig=plt.figure(figsize=(10,7))
# ax1=fig.add_subplot(2,2,1)
# ax1.title.set_text('orginal')
# plt.imshow(img)

# ax2=fig.add_subplot(2,2,2)
# ax2.title.set_text('unifom noise')
# plt.imshow(uniformNoise(img))

# ax3=fig.add_subplot(2,2,3)
# ax3.title.set_text('Gaussian noise')
# plt.imshow(GaussianNoise(img))

# ax4=fig.add_subplot(2,2,4)
# ax4.title.set_text('salt & pepper noise')
# plt.imshow(salt_pepperNoise(img))
# plt.show()

# smoothing filter
# nosyImg=salt_pepperNoise(img)
# nosyImg=cv2.imread("Images\inputs\original2.png") 

def avr_filter(size=3):
    img = read_rgb()
    x,y,_ =img.shape
    
    r,g,b=cv2.split(img)
    collored_img = [r,g,b]
    for c in range(3):
        filterColor= np.zeros((x-size+1,y-size+1),dtype=np.float64)
        for i in range(0,x-size+1):
            for j in range(0,y-size+1):
                filterColor[i,j]=((collored_img[c][i:size+i, j:size+j]).sum())/(size**2)
        collored_img[c] = filterColor
    
    
    img=cv2.merge(collored_img)
    img = img*225
    path = saveImg_unique(img, "./static/img/output/", rgb=False)
    return path

def med_filter(size=3):
    img = read_rgb()
    x,y,_ =img.shape
    r,g,b=cv2.split(img)
    collored_img = [r,g,b]
    for c in range(3):
        filterColor= np.zeros((x-size+1,y-size+1),dtype=np.float64)
        for i in range(0,x-size+1):
            for j in range(0,y-size+1):
                temp=(collored_img[c][i:size+i, j:size+j])
                temp=np.concatenate(temp )
                temp=np.sort(temp)
                index=np.int64((len(temp)+1)/2)
                filterColor[i][j]=temp[index]
        collored_img[c] = filterColor
    
    img=cv2.merge(collored_img)
    img = img*225
    path = saveImg_unique(img, "./static/img/output/", rgb=False)
    return path
def gass_filter(img=None,size = 5,sigma=1,ret= "path",rgb=True):
        if img is None:
            img = read_rgb()
        if rgb:
            x,y,_= img.shape
            r,g,b=cv2.split(img)
            collored_img = [r,g,b]
            # i=np.floor(size-1/2)
            i=(size-1)/2
            kX= np.arange(-i,i+1)
            kY= np.arange(-i,i+1)
            X,Y= np.meshgrid(kX,kY)
            g=np.exp(-((X**2+Y**2)/(2*sigma**2)))
            g=g/g.sum()
            for c in range(3):
                    filterColor= np.zeros((x-size+1,y-size+1),dtype=np.float64)
                    for i in range(0,x-size+1):
                        for j in range(0,y-size+1):
                                temp= (collored_img[c][i:size+i, j:size+j])*g
                                filterColor[i][j]=temp.sum()
                    collored_img[c] = filterColor
                    # print(filterColor)
            img=cv2.merge(collored_img)
            img = img*225
        else:
            x,y= img.shape
            i=(size-1)/2
            kX= np.arange(-i,i+1)
            kY= np.arange(-i,i+1)
            X,Y= np.meshgrid(kX,kY)
            g=np.exp(-((X**2+Y**2)/(2*sigma**2)))
            g=g/g.sum()
            
            filterImg= np.zeros((x-size+1,y-size+1),dtype=np.float64)
            for i in range(0,x-size+1):
                for j in range(0,y-size+1):
                        temp= (img[i:size+i, j:size+j])*g
                        filterImg[i][j]=temp.sum()
            img = filterImg

            img = img*225
        if ret == "path":
            path = saveImg_unique(img, "./static/img/output/", rgb=False)
            return path
        elif ret == "img":
            return img


sobel_operator_x =np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
prewitt_operator_x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_robert = [[1, 0], [0, -1]]


# sobel_filter and prewitt_filter
def filter(image = None, kernal="sobel", r= "img"):
    if image is None:
        image = read_rgb(gray=True, normalize=False)
    if kernal == "sobel":
        kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif kernal == "prewitt":
        kernal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # Create a new image with the same size as the original
    new_image = np.zeros(image.shape)
    Ix = np.zeros(image.shape)
    Iy = np.zeros(image.shape)
    # Define the Sobel filter kernels
    kernel_x = kernal
    kernel_y = np.transpose(kernal)
    # Iterate over each pixel in the image
    for i in range(image.shape[0]-2):
        for j in range(image.shape[1]-2):
            
            # Calculate the convolution of the image with each of the Sobel filters 
            gx = np.multiply(image[i:i+3, j:j+3], kernel_x ) 
            
            gy = np.multiply(image[i:i+3, j:j+3], kernel_y ) 
            gx_summed =np.sum(gx)
            Ix[i][j]=gx_summed
            gy_summed =np.sum(gy)
            Iy[i][j]=gy_summed
            # Calculate the magnitude of the gradient using Pythagoras' theorem 
            magnitude = np.sqrt(gy_summed**2+ gx_summed**2)
            # Set each pixel in the new image to be equal to its magnitude 
            if  magnitude > 0 and magnitude<255 :
                new_image[i][j]=magnitude
            elif magnitude >255 :
                new_image[i][j]=350

    if r == "img":
        path = saveImg_unique(new_image, "./static/img/output/", rgb=False, save_on_current = False)
        return path
    elif r == "xy":
        return Ix , Iy

#robert_filter
def filter_robert(kernel = [[1, 0], [0, -1]]):
    image = read_rgb(gray=True,normalize = False)
    # Create a new image with the same size as the original
    new_image = np.zeros(image.shape)
    # Define the Sobel filter kernels
    kernel_x = kernel
    kernel_y = np.transpose(kernel)
    # Iterate over each pixel in the image
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            
            # Calculate the convolution of the image with each of the Sobel filters 
            gx = np.multiply(image[i:i+2, j:j+2], kernel_x ) 
            gy = np.multiply(image[i:i+2, j:j+2], kernel_y ) 
            gx_summed =np.sum(gx)
            gy_summed =np.sum(gy)
            # Calculate the magnitude of the gradient using Pythagoras' theorem 
            magnitude = np.sqrt(gy_summed**2+ gx_summed**2)
            # Set each pixel in the new image to be equal to its magnitude 
            if  magnitude > 0 and magnitude<255 :
                new_image[i][j]=magnitude
            elif magnitude >255 :
                new_image[i][j]=350
    
    path = saveImg_unique(new_image, "./static/img/output/", rgb=False, save_on_current = False)
    return path



def non_max_suppression(img, D):
    x, y = img.shape
    suppImg = np.zeros((x, y), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, x-1):
        for j in range(1, y-1):

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                suppImg[i, j] = img[i, j]
            else:
                suppImg[i, j] = 0

    return suppImg

def cannyThr(img, lowThresholdRatio=0.01, highThresholdRatio=0.19):

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = img.max() * lowThresholdRatio

    x, y = img.shape
    res = np.zeros((x, y), dtype=np.int32)

    weak = np.int32(20)
    strong = np.int32(255)
    for i in range(x):
        for j in range(y):
            if img[i, j] >= highThreshold:
                res[i, j] = strong
            elif (img[i, j] <= highThreshold) and (img[i, j] >= lowThreshold):
                res[i, j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    x, y = img.shape
    for i in range(1, x-1):
        for j in range(1, y-1):
            if (img[i, j] == weak):

                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img

def canny(low_threshold=None,high_threshold=None,sigma=1):
    
    img = read_rgb(gray=True)
    gaussian_img = gass_filter(img,size = 9,sigma=sigma,rgb=False,ret="img")
    x , y =  filter(image=gaussian_img, kernal="sobel",r="xy")
    mag = np.hypot(x, y)
    mag = mag/mag.max()*255
    theta = np.arctan2(y, x)
    sup = non_max_suppression(mag, theta)
    res, week, str = cannyThr(sup)
    finalImg = hysteresis(res, week, str)
    path = saveImg_unique(finalImg, "./static/img/output/", rgb=False, save_on_current = False)
    return path
