import numpy as np
from read_images import read_folder
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
from scipy.optimize import curve_fit

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
    convolved = convolve(img, kernel, mode='valid')
    return convolved

def get_mesh_coordinates(max_x, max_y, dx, dy):
    x = np.arange(np.int32(dx), max_x-np.int32(dx), np.int32(dx))
    y = np.arange(np.int32(dy), max_y-np.int32(dy), np.int32(dy))
    X,Y = np.meshgrid(x,y)
    return X, Y

def func_fit(x, a, b):
    return a*x+  b

def dfunc_fit(x, a, b):
    return a

def deriv_on_grid(data, mesh_x, mesh_y):
    # derivs in y direction
    velocities_x = []
    for i in range(mesh_x.shape[0]):
        aux=[]
        for j in range(mesh_x.shape[1]):
            # saco la serie temporal para un x fijo y la suavizo
            denoised = gaussian_denoising(get_pixel_in_time(data, x_idx=mesh_x[i,j], y_idx=mesh_y[i,j]), kernel_size=11)
            # saco los picos en el tiempo
            peaks, props = find_peaks(denoised, distance = 10, prominence = 0.4, width=2)
            # elijo el primer pico temporal
            aux.append(peaks[0])
            plt.plot(denoised+j)
            plt.plot(peaks[0],j+1, 'rx')
        plt.show()
        aux = np.array(aux)
        clean_mask = np.abs(aux - aux.mean()) < 2*aux.std()
        aux = aux[clean_mask]
        popt,_ = curve_fit(func_fit, aux, mesh_x[i,clean_mask])
        velocities_x.append(popt[0])
    velocities_x =np.array(velocities_x)
    plt.imshow(data[120].T, origin='lower')
    for i in range(mesh_x.shape[0]):
        plt.quiver(mesh_x[i,:], mesh_y[i,:], velocities_x[i], 0,color = 'y')
    plt.show()

    velocities_y = []
    for i in range(mesh_y.shape[1]):
        aux=[]
        for j in range(mesh_y.shape[0]):
            denoised = gaussian_denoising(get_pixel_in_time(data, x_idx=mesh_x[j,i], y_idx=mesh_y[j,i]), kernel_size=11)
            peaks, props = find_peaks(denoised, distance = 10, prominence = 0.4, width=2)
            aux.append(peaks[0])
            plt.plot(denoised+j)
            plt.plot(peaks[0],j+1, 'x')
        plt.show()
        aux = np.array(aux)
        clean_mask = np.abs(aux - aux.mean()) < 2*aux.std()
        aux = aux[clean_mask]
        popt,_ = curve_fit(func_fit, aux, mesh_y[clean_mask,i])
        velocities_y.append(popt[0])
    velocities_y =np.array(velocities_y)
    plt.imshow(data[120].T, origin='lower')
    for i in range(mesh_x.shape[1]):
        plt.quiver(mesh_x[:,i], mesh_y[:,i], 0,velocities_y[i],color = 'y')
    plt.show()

    VX, VY = np.meshgrid(velocities_x, velocities_y)
    plt.imshow(data[120].T, origin='lower')
    plt.quiver(mesh_x, mesh_y, VX, VY,color = 'y')
    plt.show()
    
    return VX, VY

if __name__ == '__main__':
    raw_imgs = read_folder('./img_fase', kind='fase')
    NX = raw_imgs[1].shape[0]
    NY = raw_imgs[1].shape[1]
    derivs = {}
    for k in range(2,len(list(raw_imgs.keys()))):
        derivs[k] = raw_imgs[k]-raw_imgs[k-1]
        mask = derivs[k] > 150
        derivs[k][mask] = 0

    xses, yses = get_mesh_coordinates(max_x= NX, max_y= NY, dx= 70, dy= 70)
    deriv_on_grid(derivs, xses, yses)