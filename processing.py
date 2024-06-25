import numpy as np
from read_images import read_folder
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks

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

def gaussian_border_detection(img, kernel_size=3, sign = 1):
    x = np.linspace(-1,1,kernel_size)
    kernel = sign*x*np.exp(-(x**2)/0.25**2)/(2*np.pi*0.25)
    convolved = convolve(img, kernel, mode='same')
    return convolved

def get_mesh_coordinates(max_x, max_y, dx, dy):
    x = np.arange(np.int32(dx), max_x-np.int32(dx), np.int32(dx))
    y = np.arange(np.int32(dy), max_y-np.int32(dy), np.int32(dy))
    X,Y = np.meshgrid(x,y)
    return X, Y

if __name__ == '__main__':
    raw_imgs = read_folder('./img_fase', kind='fase')
    NX = raw_imgs[1].shape[0]
    NY = raw_imgs[1].shape[1]
    derivs = {}
    for k in range(2,len(list(raw_imgs.keys()))):
        derivs[k] = raw_imgs[k]-raw_imgs[k-1]
        mask = derivs[k] > 150
        derivs[k][mask] = 0


    fig, axs = plt.subplots(1,2)
    frame = 100
    axs[0].set_title(f'frame{frame}')
    axs[0].imshow(derivs[frame].T)
    print(derivs[frame].shape)

    # sobre eje
    xses, yses = get_mesh_coordinates(max_x= NX, max_y= NY, dx= 50, dy= 50)
    axs[0].scatter(xses, yses, marker='x', color='r', s=5)
    for i in range(xses.shape[1]):
        denoised = gaussian_denoising(get_pixel_in_time(derivs, x_idx=xses[-1,i], y_idx=yses[-1,i]), kernel_size=21)

        peaks, props = find_peaks(denoised, distance = 10, prominence = 0.5, width=2)
        axs[1].plot(denoised+i, alpha = 0.5)
        axs[1].vlines(peaks, i, props["prominences"]+i, ls='dotted')
        axs[1].hlines(y=props["prominences"]+i, xmin=props["left_ips"],xmax=props["right_ips"])

    plt.show()