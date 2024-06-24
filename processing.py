import numpy as np
from read_images import read_folder
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    raw_imgs = read_folder('./img_modulo', kind='modulo')

    plt.plot(get_pixel_in_time(raw_imgs, x_idx=10, y_idx=10), alpha = 0.5)
    plt.plot(get_pixel_in_time(raw_imgs, x_idx=10, y_idx=55), alpha = 0.5)
    plt.show()

    plt.plot(correlate_signals(get_pixel_in_time(raw_imgs, x_idx=10, y_idx=10), get_pixel_in_time(raw_imgs, x_idx=10, y_idx=55)))
    plt.show()