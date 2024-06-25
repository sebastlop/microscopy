import numpy as np
from read_images import read_folder
import matplotlib.pyplot as plt
from scipy.signal import convolve

DELTA_TIME = 0.1


def get_pixel_in_time(img_batch: dict, x_idx = 0, y_idx = 0):
    idx_min = np.min(list(img_batch.keys()))
    idx_max = np.max(list(img_batch.keys()))
    pixel_time = []
    for i in range(idx_min, idx_max+1, 1):
        pixel_time.append(img_batch[i][x_idx, y_idx])
    return np.array(pixel_time)

def correlate_signals(s1, s2):
    correlog = np.correlate(s1,s2,mode='valid')[0]
    for d in range(1,100):
        correlog = np.hstack([correlog,np.correlate(s1[d:],s2[:-d],mode='valid')[0]])
    return correlog

def gaussian_denoising(img, kernel_size=41):
    x = np.linspace(-1,1,kernel_size)
    kernel = np.exp(-(x**2)/0.25**2)/(2*np.pi*0.25)
    convolved = convolve(img, kernel, mode='same')
    convolved= np.minimum((convolved-convolved.min())/(convolved.max()-convolved.min()),1)
    convolved = np.maximum(convolved,0)
    return convolved

def gaussian_border_detection(img, kernel_size=41):
    x = np.linspace(-1,1,kernel_size)
    kernel = x*np.exp(-(x**2)/0.25**2)/(2*np.pi*0.25)
    convolved = convolve(img, kernel, mode='same')
    return convolved

if __name__ == '__main__':
    raw_imgs = read_folder('./img_fase', kind='fase')
    for idx in range(100,130):
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(raw_imgs[idx])
        axs[1].imshow(raw_imgs[idx+1]-raw_imgs[idx])
        plt.show()
    exit()



    plt.plot(get_pixel_in_time(raw_imgs, x_idx=13, y_idx=250), alpha = 0.5)
    plt.plot(get_pixel_in_time(raw_imgs, x_idx=16, y_idx=250), alpha = 0.5)
    plt.plot(get_pixel_in_time(raw_imgs, x_idx=19, y_idx=250), alpha = 0.5)

    plt.show()