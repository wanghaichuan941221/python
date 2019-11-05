import numpy as np
import matplotlib.pyplot as plt

def fftwave(u,v,sz):
    Fhat = np.zeros(sz)
    Fhat[u,v]=1

    F = np.fft.ifft2(Fhat)
    # Fabsmax = np.max(np.abs(F[:]))
    # print(Fabsmax.shape)

    Fhat_shift = np.fft.fftshift(Fhat)

    plt.subplot(3,2,1),plt.imshow(Fhat,cmap="gray"),plt.title('Fhat')
    plt.subplot(3,2,2),plt.imshow(Fhat_shift,cmap="gray"),plt.title('Fhat shift')
    plt.subplot(3,2,3),plt.imshow(np.real(F),cmap="gray"),plt.title('real part of image')
    plt.subplot(3,2,4),plt.imshow(np.imag(F),cmap="gray"),plt.title('imag part of image')
    plt.subplot(3,2,5),plt.imshow(np.abs(F),cmap="gray"),plt.title('abs value of image')
    plt.subplot(3,2,6),plt.imshow(np.angle(F),cmap="gray"),plt.title('wavelength of image')

    plt.show()


fftwave(9,9,(128,128))
