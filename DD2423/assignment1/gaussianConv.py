import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import fftpack

def build_filter():
    # X = np.linspace(-4,4,1000)
    # Y = np.linspace(-4,4,1000)
    # x,y = np.meshgrid(X,Y)
    # kernel = 0.001*np.exp(-10*(x**2+y**2))
    t = np.linspace(-4, 4,1000)
    bump = np.exp(-10*t**2)
    bump /= np.trapz(bump)
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    
    #plot gaussian kernel
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_wireframe(x,y,gauss,color='b')
    # ax.plot_surface(x,y,kernel,cmap='viridis')
    # plt.show()

    return kernel


def image_conv(img,kernel=build_filter()):
    # flip the kernel, gaussian is symmetric, so this one is not necessary
    kernel1 = cv2.flip(kernel,-1)
    # corelation function filter2D
    res = cv2.filter2D(img,-1,kernel)
    
    plt.subplot(1,2,1),plt.imshow(img,cmap="gray")
    plt.subplot(1,2,2),plt.imshow(res,cmap="gray")
    plt.show()

def image_fft(img,kernel=build_filter()):
    # kernel_ft = fftpack.fft2(kernel, shape=img.shape, axes=(0, 1))
    # img_ft = fftpack.fft2(img, axes=(0, 1))

    kernel_ft = np.fft.fft2(kernel)
    # kernel_ft = np.resize(kernel_ft,img.shape)
    # img_ft = np.fft.fft2(img)

    # kernel_ft = np.fft.fftshift(kernel_ft)
    # img_ft = np.fft.fftshift(img_ft)
    
    # img2_ft = kernel_ft * img_ft
    # img2 = np.fft.ifft2(img2_ft).real
        
    # plt.subplot(1,2,1),plt.imshow(img,cmap="gray")
    # plt.subplot(1,2,2),plt.imshow(img2,cmap="gray")
    # plt.show()

    # return img2


# F = np.concatenate(( np.zeros((60,128)),np.ones((8,128)),np.zeros((60,128)) ))
# G = np.concatenate(( np.zeros((48,128)),np.ones((32,128)),np.zeros((48,128)) ))
# img = F.dot(G.T)
img = cv2.imread("apple_gray.jpg",0)
# image_conv(img)
img2 = image_fft(img)
