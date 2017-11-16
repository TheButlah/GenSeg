from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os

from scipy import misc, io
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2lab, lab2rgb


def gen_occupancy_grid(x, lower_left, upper_right, divisions):
    output = np.zeros(np.append(divisions, 1))
    lengths = upper_right - lower_left
    intervals = lengths / divisions
    offsets = x - lower_left
    indices = np.floor(offsets / intervals)
    indices = indices.astype(int)
    for row in indices:
        if np.sum(row >= np.zeros([1, 3])) == 3 and np.sum(row < divisions) == 3:
            output[row[0], row[1], row[2], 0] = 1
    return output


def gen_label_occupancy_grid(x, lower_left, upper_right, divisions, num_classes):
    output = np.zeros(np.append(divisions, num_classes))
    lengths = upper_right - lower_left
    intervals = lengths / divisions
    offsets = x - np.append(lower_left, 0)
    indices = np.floor(offsets / np.append(intervals, 1))
    indices = indices.astype(int)
    for row in indices:
        if np.sum(row[:3] >= np.zeros([1, 3])) == 3 and np.sum(row[:3] < divisions) == 3:
            output[row[0], row[1], row[2], row[3]] += 1
    output = np.argmax(output, -1)
    return output


class DataReader(object):
    def __init__(self, path, image_shape, lower_left, upper_right, divisions, num_classes):
        self._image_shape = image_shape
        self._lower_left = lower_left
        self._upper_right = upper_right
        self._divisions = divisions
        self._num_classes = num_classes
        self._path = os.path.abspath(path)
        self._image_data = self.get_filenames(path + '/image_data/testing/')
        self._image_labels = self.get_filenames(path + '/image_labels/testing/')
        self._velodyne_data = self.get_filenames(path + '/velodyne_data/testing/')
        self._velodyne_labels = self.get_filenames(path + '/velodyne_labels/testing/')

    def get_filenames(self, path):
        data_paths = os.listdir(path)
        data_paths = sorted(data_paths)
        data_paths = [path + data_path for data_path in data_paths]
        filenames = []
        for data_path in data_paths:
            _filenames = os.listdir(data_path)
            _filenames = sorted(_filenames)
            _filenames = [data_path + '/' + filename for filename in _filenames]
            filenames += _filenames
        return filenames

    def get_image_data(self):
        h, w, c = self._image_shape
        shape = (len(self._image_labels), h // 2, w // 2, c)
        img_data_loc = make_path('processed/img_data.npy')

        if os.path.exists(img_data_loc):
            image_data = np.load(img_data_loc)
            return image_data

        image_data = np.empty(shape)
        k = 0
        for filename in self._image_data:
            image = normalize_img(misc.imread(filename))  # Fix brightness and convert to lab colorspace
            image_data[k, :, :, :] = image[0:h:2, 0:w:2, 0:c]
            k += 1
        np.save(img_data_loc, image_data)
        return image_data

    def get_image_labels(self):
        h, w, _ = self._image_shape
        shape = (len(self._image_labels), h // 2, w // 2)
        img_labels_loc = make_path('processed/img_labels.npy')

        if os.path.exists(img_labels_loc):
            label_data = np.load(img_labels_loc)
            return label_data

        label_data = np.empty(shape)
        k = 0
        for filename in self._image_labels:
            label = io.loadmat(filename)
            label = label['truth']
            label_data[k, :, :] = label[0:h:2, 0:w:2]
            k += 1
        np.save(img_labels_loc, label_data)
        return label_data

    def get_velodyne_data(self):
        shape = np.append(self._divisions, 1)
        shape = np.insert(shape, 0, len(self._velodyne_data))
        vel_data_loc = make_path('processed/vel_data.npy')

        if os.path.exists(vel_data_loc):
            velo_data = np.load(vel_data_loc)
            return velo_data

        velo_data = np.empty(shape)
        k = 0
        for filename in self._velodyne_data[:1]:
            velo = np.fromfile(filename, dtype='float32')
            velo = np.reshape(velo, [-1, 4])
            #velo = np.reshape(velo, [4, -1])
            #velo = np.transpose(velo)
            velo = velo[:, 0:3]
            velo = gen_occupancy_grid(velo, self._lower_left, self._upper_right, self._divisions)
            velo_data[k, :, :, :, :] = velo
            k += 1
            print(k)
        np.save(vel_data_loc, velo_data)
        return velo_data

    def get_velodyne_labels(self):
        shape = np.insert(self._divisions, 0, len(self._velodyne_data))
        vel_labels_loc = make_path('processed/vel_labels.npy')

        if os.path.exists(vel_labels_loc):
            label_data = np.load(vel_labels_loc)
            return label_data

        label_data = np.empty(shape)
        k = 0
        for (data_filename, label_filename) in zip(self._velodyne_data, self._velodyne_labels):
            velo = np.fromfile(data_filename, dtype='float32')
            velo = np.reshape(velo, [-1, 4])
            #velo = np.reshape(velo, [4, -1])
            #velo = np.transpose(velo)
            velo = velo[:, 0:3]
            label = io.loadmat(label_filename)
            label = label['truth']
            velo = np.concatenate([velo, label], 1)
            velo = gen_label_occupancy_grid(velo, self._lower_left, self._upper_right, self._divisions, self._num_classes)
            label_data[k, :, :, :] = velo
            k += 1
            print(k)
        np.save(vel_labels_loc, label_data)
        return label_data


def original_to_label(original):
    return {
        3: 1,  # road
        5: 2,  # sidewalk
        6: 3,  # car
        7: 4,  # pedestrian
        8: 5  # cyclist
    }.get(original, 0)  # unkown


def label_to_original(label):
    return {
        1: 3,  # road
        2: 5,  # sidewalk
        3: 6,  # car
        4: 7,  # pedestrian
        5: 8  # cyclist
    }.get(label, 0)

def variance_color(original):
    o = original
    return [255 - o * 255, 255 - o * 255, 255 - o * 255]


def get_color(original):  # function to map ints to RGB array
    return {
        1: [153, 0, 0],  # building
        2: [0, 51, 102],  # sky
        3: [160, 160, 160],  # road
        4: [0, 102, 0],  # vegetation
        5: [255, 228, 196],  # sidewalk
        6: [255, 200, 50],  # car
        7: [255, 153, 255],  # pedestrian
        8: [204, 153, 255],  # cyclist
        9: [130, 255, 255],  # signage
        10: [193, 120, 87],  # fence
    }.get(original, [0, 0, 0])  # Unknown


def normalize_img(img):
    hist = equalize_adapthist(img) * 255  # equalize_adapthist turns into floats from [0,1]
    hist = hist.astype(np.uint8, copy=False)
    lab = rgb2lab(hist)
    return lab


def index_to_real(coord, ll, ur, divisions):
    interval = (ur-ll)/divisions
    return coord*interval+ll


def real_to_index(coord, ll, ur, divisions):
    interval = (ur-ll)/divisions
    return np.floor_divide((coord - ll), interval)


def make_path(file):
    """Makes the directories required for `file` to reside in.

    Args:
        file: A string giving the path to the file.
    Returns:
        The absolute path of the original file.
    """
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.abspath(file)

# dr = DataReader('/home/vdd6/Desktop/gen_seg_data', (374, 1238, 3))
# res = dr.get_image_data()
# res = dr.get_image_labels()
