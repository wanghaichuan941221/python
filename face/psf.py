import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D
from skimage import color, data, restoration
from astropy.convolution import Gaussian2DKernel

def PSF(img1,img):
    min_num,sigma1= 100,0
    num=np.zeros(60)
    count=np.zeros(60)
    for i in range(1,60):
        sigma = 0.1*i
        # count[i-5]=sigma
        psf = Gaussian2DKernel(sigma)
        img_conv=convolve2d(img,psf,'same')
        diff = np.abs(img_conv-img1).mean()
        num[i-1] = diff
        if diff < min_num:
            min_num=diff
            sigma1 = sigma
    # return num
    return count,num,sigma1,min_num

def visulization_3d(num):
    x=y=np.linspace(1,10,10)
    # y=np.linspace(0,29,30)
    # x=np.linspace(0,49,50)
    x,y=np.meshgrid(x,y)
    x,y=0.001*x,0.001*y
    # print(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(x,y,num)
    ax.set_xlabel("std1")
    ax.set_ylabel("std2")
    ax.set_zlabel("average difference")
    # plt.xlabel('std1*0.001'),plt.ylabel('std2*0.001')
    plt.show()

def FFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

def noisy(img_conv,img1):
    # if noise_typ == "gauss":
    diff0 = 1000
    row,col = img.shape
    m1,var1,para1=0,0,0
    # num = np.zeros((60,50))
    for m in range(0,60):
        for var in range(0,50):
            sigma = var**0.5
            gauss = 0.16*np.random.normal(m,sigma,(row,col))
            gauss = gauss.reshape(row,col)
            diff1=np.abs(img1-img_conv-gauss).mean()
            if diff1< diff0:
                diff0 = diff1
                var1,m1 = var,m
    # print(m1,var1)
    return m1,var1

def image_sharpness(img,img1):
    s1,s2,a1,diff1=0,0,0,100.
    num = np.zeros((10,10,10))
    for std1 in range(0,10):
        for std2 in range(0,10):
            for a in range(0,10):
                g1 = np.random.normal(0,std1*0.001,img.shape)
                g2 = np.random.normal(0,std2*0.001,img.shape)
                img_sharp = img+a*0.1*convolve2d(img,g1-g2,'same')
                diff = np.abs(img_sharp-img1).mean()
                num[std1,std2,a]=diff
                if diff < diff1:
                    diff1 = diff
                    s1,s2,a1=std1,std2,a
    return num,a1,s1,s2,diff1
    # return s1,s2,a1
    # return diff1


def img_plot(img1,img2):
    plt.subplot(2,2,1),plt.imshow(img1),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(img2),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(FFT(img1)),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(FFT(img2)),plt.xticks([]),plt.yticks([])
    plt.show()

img1 = cv2.imread('img'+str(1)+'.JPG')
img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

diff_sharp = np.zeros(10)
diff_psf = np.zeros(10)
diff_noise = np.zeros(10)
dr = np.zeros(10)
for i in range(2,12):
    img = cv2.imread('img'+str(i)+'.JPG')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    dr[i-2]=img1.shape[0]/img.shape[0]
    img = cv2.resize(img,(67,74))

    count,num,sigma,min_num=PSF(img1,img)
    diff_psf[i-2] = min_num
    psf = Gaussian2DKernel(sigma)
    img_conv = convolve2d(img,psf,'same')

    num1,a1,s1,s2,diff1 = image_sharpness(img_conv,img1)
    diff_sharp[i-2] = diff1
    g1 = np.random.normal(0,s1*0.001,img.shape)
    g2 = np.random.normal(0,s2*0.001,img.shape)
    img_sharp = img_conv+a1*0.1*convolve2d(img,g1-g2,'same')

    img_diff = FFT(img1)-FFT(img_sharp)
    fft_diff_min = 100
    sigma1,sig1 = 0,0
    # # plt.subplot(1,11,1),plt.imshow(img1),plt.xticks([]),plt.yticks([])
    for sigma in range(1,100):
        for sig in range(1,100):
            guass_noise = np.random.normal(0,0.001*sigma,img_diff.shape)
            guass_noise = gaussian_filter(guass_noise,0.01*sig)
            fft_diff = np.abs(FFT(guass_noise)-img_diff).mean()
            # f100[sigma-1,sig-1] = fft_diff
            # s100[sigma-1] = 0.001*sigma
            # s200[sig-1] = 0.01*sig
            if fft_diff < fft_diff_min:
                fft_diff_min = fft_diff
                diff_noise[i-2] = fft_diff
                # sigma1 = sigma
                # sig1 = sig
    # g_noise = np.random.normal(0,sigma1*0.01,img1.shape)
    # g_noise = gaussian_filter(g_noise,0.01*sig1)
    # img_final = img_sharp+g_noise
    # cv2.imwrite("final/img"+str(i)+'.JPG',img_final)
#     plt.subplot(1,11,i),plt.imshow(img_sharp+g_noise),plt.xticks([]),plt.yticks([])
# plt.show()


# img = cv2.imread('img'+str(10)+'.JPG')
# img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img = cv2.resize(img,(67,74))
# num = PSF(img1,img)
# count,num,sigma,min_num=PSF(img1,img)
# psf = Gaussian2DKernel(sigma)
# img_conv = convolve2d(img,psf,'same')
#
# num1,a,s1,s2,diff1= image_sharpness(img_conv,img1)
# g1 = np.random.normal(0,s1*0.001,img.shape)
# g2 = np.random.normal(0,s2*0.001,img.shape)
# img_sharp = img_conv+a*0.1*convolve2d(img,g1-g2,'same')
#
# img_diff = FFT(img1)-FFT(img_sharp)
# fft_diff_min = 100
# fft_diff_1 = np.zeros([100,100])
# for sigma in range(1,100):
#     for sigma1 in range(1,100):
#         guass_noise = np.random.normal(0,0.01*sigma,img_diff.shape)
#         guass_noise = gaussian_filter(guass_noise,0.01*sigma1)
#         fft_diff = np.abs(FFT(guass_noise)-img_diff).mean()
#         fft_diff_1[sigma-1,sigma1-1]=fft_diff
        # if fft_diff < fft_diff_min:
        #     fft_diff_min = fft_diff

#             print(fft_diff_min,sigma,sigma1)
#
# g_noise = np.random.normal(0,sigma1*0.01,img1.shape)
# g_noise = gaussian_filter(g_noise,0.01*sig1)
