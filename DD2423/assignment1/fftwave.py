import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
import cv2

########################################
## basic function
########################################
def fftwave(u,v,sz):
    Fhat = np.zeros(sz)
    Fhat[u,v]=1

    F = np.fft.ifft2(Fhat)

    Fhat_shift = np.fft.fftshift(Fhat)

    plt.subplot(3,2,1),plt.imshow(Fhat,cmap="gray"),plt.title('Fhat')
    plt.subplot(3,2,2),plt.imshow(Fhat_shift,cmap="gray"),plt.title('Fhat shift')
    plt.subplot(3,2,3),plt.imshow(np.real(F),cmap="gray"),plt.title('real part of image')
    plt.subplot(3,2,4),plt.imshow(np.imag(F),cmap="gray"),plt.title('imag part of image')
    plt.subplot(3,2,5),plt.imshow(np.abs(F),cmap="gray"),plt.title('abs value of image')
    plt.subplot(3,2,6),plt.imshow(np.angle(F),cmap="gray"),plt.title('wavelength of image')

    plt.show()

def linearity():
    F = np.concatenate(( np.zeros((56,128)),np.ones((16,128)),np.zeros((56,128)) ))
    G = F.transpose()
    H = F + 2*G

    Fhat = np.fft.fft2(F)
    Ghat = np.fft.fft2(G)
    Hhat = np.fft.fft2(H)

    plt.subplot(2,2,1),plt.imshow(np.log(1+abs(np.fft.fftshift(Fhat))),cmap="gray")
    plt.subplot(2,2,2),plt.imshow(np.log(1+abs(np.fft.fftshift(Ghat))),cmap="gray")
    plt.subplot(2,2,3),plt.imshow(np.log(1+abs(Hhat)),cmap="gray")
    plt.subplot(2,2,4),plt.imshow(np.log(1+abs(np.fft.fftshift(Hhat))),cmap="gray")
    plt.show()

def Multiplication():
    F = np.concatenate(( np.zeros((56,128)),np.ones((16,128)),np.zeros((56,128)) ))
    G = F.transpose()

    # plot F * G
    plt.subplot(1,2,1),plt.imshow(F.dot(G),cmap="gray")
    
    plt.subplot(1,2,2),plt.imshow(10* np.log (1+np.abs(np.fft.fftshift(np.fft.fft2(F.dot(G))))),cmap="gray")
    plt.show()

def scaling():
    # compare the result with multiplication
    F = np.concatenate(( np.zeros((60,128)),np.ones((8,128)),np.zeros((60,128)) ))
    G = np.concatenate(( np.zeros((48,128)),np.ones((32,128)),np.zeros((48,128)) ))

    plt.subplot(1,2,1),plt.imshow(F.dot(G.T),cmap="gray")
    plt.subplot(1,2,2),plt.imshow(10* np.log (1+np.abs(np.fft.fftshift(np.fft.fft2(F.dot(G.T))))),cmap="gray")
    plt.show()

def rotation(angle=30):
    F = np.concatenate(( np.zeros((60,128)),np.ones((8,128)),np.zeros((60,128)) ))
    G = np.concatenate(( np.zeros((48,128)),np.ones((32,128)),np.zeros((48,128)) ))
    img = F.dot(G.T)
    # rotate image with scipy.ndimage
    # img_30 = ndimage.rotate(img,angle,reshape=False)
    
    # rotate image with opencv
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img_30 = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    img_hat = np.fft.fft2(img)
    img_hat_30 = np.fft.fft2(img_30)

    plt.subplot(2,2,1),plt.imshow(img_30,cmap="gray"),plt.title('rotate image')
    plt.subplot(2,2,2),plt.imshow(np.log(1+abs(np.fft.fftshift(img_hat_30))),cmap="gray"),plt.title('amplitute of fft image')
    plt.subplot(2,2,3),plt.imshow(img,cmap="gray"),plt.title('original image')
    plt.subplot(2,2,4),plt.imshow(np.log(1+abs(np.fft.fftshift(img_hat))),cmap="gray"),plt.title('amplitute of fft image')

    plt.show()
  

    



# call fftwave 
# fftwave(5,9,(128,128))

# linearity
# linearity()

# Multiplication
# Multiplication()

# scaling
# scaling()

# rotation
rotation()