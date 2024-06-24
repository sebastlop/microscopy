import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import os

def read_folder(folder, kind = 'modulo'):
    images_raw = {}
    filelist = os.listdir(folder)
    if kind == 'modulo':
        sep = '-'
        for f in filelist:
            if f.find(kind) != -1:
                idx = f.find(sep)
                fig = img.imread(os.path.join(folder, f))
                images_raw[int(f[:idx])] = fig

    elif kind == 'fase':
        idx = -8
        for f in filelist:
            if f.find(kind) != -1:
                fig = img.imread(os.path.join(folder, f))
                images_raw[int(f[:idx])] = fig
    return images_raw


if __name__ == '__main__':
    read_folder('./img_fase', kind='fase')