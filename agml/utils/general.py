# Copyright 2021 UC Davis Plant AI and Biophysics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from math import pi, floor
from scipy import signal

import numpy as np


# Represents an empty object, but allows passing `None`
# as an independent object in certain cases.
NoArgument = object()


def placeholder(obj):
    """Equivalent of lambda x: x, but enables pickling."""
    return obj


def to_camel_case(s):
    """Converts a given string `s` to camel case."""
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "") # noqa
    return ''.join(s)


def resolve_list_value(val):
    """Determines whether a list contains one or multiple values."""
    if len(val) == 1:
        return val[0]
    return val


def resolve_tuple_values(*inputs, custom_error = None):
    """Determines whether `inputs[0]` contains two values or
    they are distributed amongst the values in `inputs`. """
    if isinstance(inputs[0], (list, tuple)) and all(c is None for c in inputs[1:]):
        if len(inputs[0]) != len(inputs):
            # special case for COCO JSON
            if len(inputs) == 3 and len(inputs[0]) == 2 and isinstance(inputs[0][1], dict):
                return inputs[0][0], inputs[0][1]['bbox'], inputs[0][1]['category_id']
            if custom_error is not None:
                raise ValueError(custom_error)
            else:
                raise ValueError(
                    f"Expected either a tuple with {len(inputs)} values "
                    f"or {len(inputs)} values across two arguments.")
        else:
            return inputs[0]
    return inputs


def resolve_tuple(sequence):
    """Resolves a sequence to a tuple."""
    if isinstance(sequence, np.ndarray):
        sequence = sequence.tolist()
    return tuple(i for i in sequence)


def has_nested_dicts(obj):
    """Returns whether a dictionary contains nested dicts."""
    return any(isinstance(i, dict) for i in obj.values())


def as_scalar(inp):
    """Converts an input value to a scalar."""
    if isinstance(inp, (int, float)):
        return inp
    if np.isscalar(inp):
        return inp.item()
    if isinstance(inp, np.ndarray):
        return inp.item()
    from agml.backend.tftorch import torch
    if isinstance(inp, torch.Tensor):
        return inp.item()
    from agml.backend.tftorch import tf
    if isinstance(inp, tf.Tensor):
        return inp.numpy()
    raise TypeError(f"Unsupported variable type {type(inp)}.")


def scalar_unpack(inp):
    """Unpacks a 1-d array into a list of scalars."""
    return [as_scalar(item) for item in inp]


def is_array_like(inp):
    """Determines if an input is a np.ndarray, torch.Tensor, or tf.Tensor."""
    if isinstance(inp, np.ndarray):
        return True
    from agml.backend.tftorch import torch
    if isinstance(inp, torch.Tensor):
        return True
    from agml.backend.tftorch import tf
    if isinstance(inp, tf.Tensor):
        return True
    return False

def txt2image(path):
    """Converts txt to numpy array"""
    data = np.loadtxt(path)
    data = np.where(data > 20, 0, data) # Change background numbers to zero
    return data


def txt2image2(path):
    """Converts txt to numpy array, not consider first row"""
    data = np.loadtxt(path,skiprows=1)
    data = np.where(data > 1000, 1001, data) # Change background numbers to zero
    return data


def PlotAllViews(path, Pos):
    """Plot all the RGB images"""
    plt.rcParams['figure.figsize'] = [50, 50]
    fig, axs = plt.subplots(round(len(Pos)/2 + 0.1),2)
    fig.subplots_adjust(hspace = .1, wspace=.001)
    axs = axs.ravel()
    
    for i in range(len(Pos)):
        if i<10:
            img = cv2.imread(path + 'view0000' + str(i) + '/RGB_rendering.jpeg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
        if i>10:
            img = cv2.imread(path + 'view000' + str(i) + '/RGB_rendering.jpeg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
        axs[i].set_title('Camera view: #' + str(i),fontsize=50)

        plt.rcParams['figure.figsize'] = [50, 50]
    if len(Pos) % 2 != 0:
        fig.delaxes(axs[i+1])

def PlotAllViewsSemantic(path, Pos):
    """Plot all the Semantic segmentation images"""
    fig, axs = plt.subplots(round(len(Pos)/2 + 0.1),2)
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i in range(len(Pos)):
        # print(i)
        if i<10:
            data = txt2image(path + 'view0000' + str(i) + '/semantic_segmentation.txt')
            img = data
            axs[i].imshow(img)
            im = axs[i].pcolormesh(data, cmap='gist_rainbow')

        if i>10:
            data = txt2image(path + 'view000' + str(i) + '/semantic_segmentation.txt')
            img = data
            axs[i].imshow(img,'gray')
            axs[i].pcolormesh(data, cmap='gist_rainbow')
            axs[i].colorbar()
            axs[i].show()
    if len(Pos) % 2 != 0:
        fig.delaxes(axs[i+1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

def generate_specific_rows(filePath, userows=[]):
    """Read specific line from txt file"""
    with open(filePath) as f:
        for i, line in enumerate(f):
            if i == userows:
                return line

def PlotInstanceSegmentation(path, view, Label):
    """Plot specific instance segmentaation images"""
    L = Label

    i = view
    
    
    if i<10:
        data1 = path + 'view0000' + str(i) 
    elif i>=10:
        data1 = path + 'view000' + str(i)
    
    Path_elements = glob.glob(data1 + '/instance_segmentation_' + str(L) + '_*')
    View_size = len(Path_elements)
    print('Number of ' + str(L) +':' + str(View_size))
        
    axs = plt.axes()
    
    img = cv2.imread(path + 'view0000' + str(view) + '/RGB_rendering.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for j in range(0,View_size):                                             
        data = txt2image2(Path_elements[j])
        rectangular_label = generate_specific_rows(Path_elements[j], userows=0)
        rectangular_label = rectangular_label.replace("\n", "").split(" ")
        data2 = rectangular_label
        data2[0] = int(data2[0])
        data2[1] = int(data2[1])
        data2[2] = int(data2[2]) 
        data2[3] = int(data2[3]) 
        X_min = data2[0]
        Y_min = img.shape[0] - data2[2] - (data2[3] - data2[2])
        W = data2[1] - data2[0]
        H = data2[3] - data2[2]
        rect = patches.Rectangle((X_min, Y_min ), W , H , linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        axs.add_patch(rect)
        
        #Add instance segmentation
        A = np.zeros(img.shape) + 1001

        imgray = data[1:,1:]
        img2 = cv2.merge((imgray,imgray,imgray))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        A[Y_min:Y_min+H,data2[0]:data2[1],:] = img2
    
        for x in range(0,img.shape[0]):
            for y in range(0,img.shape[1]):
                if x > Y_min and x < Y_min+H and y > data2[0] and y < data2[1]:
                    if A[x,y,0]<1000: 
                        img[x,y,0] = 50*A[x,y,0]
                        img[x,y,1] = 20*A[x,y,0]
                        img[x,y,2] = 40*A[x,y,0]
        axs.imshow(img)
        
    plt.show()





def PlotObjectDetection(path, view, Label):
    """Plot object detection on one RGB image"""
    plt.rcParams['figure.figsize'] = [15, 15]
    axs = plt.axes()
    
    L = Label

    if view<10:
        img = cv2.imread(path + 'view0000' + str(view) + '/RGB_rendering.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs.imshow(img)
        data = np.loadtxt(path + 'view0000' + str(view) + '/rectangular_labels_' + str(L) + '.txt')

    if view>10:
        img = cv2.imread(path + 'view000' + str(view) + '/RGB_rendering.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs.imshow(img)
        data = np.loadtxt(path + 'view000' + str(view) + '/rectangular_labels_' + str(L) + '.txt')
        
    axs.set_title('Camera view: #' + str(view),fontsize=50)
    # Create a Rectangle patch
    img_shape = img.shape
    for l in range(len(data)):
        data[l][1] = data[l][1]*img.shape[1]
        data[l][3] = data[l][3]*img.shape[1]
        data[l][2] = data[l][2]*img.shape[0]
        data[l][4] = data[l][4]*img.shape[0]
        rect = patches.Rectangle(((data[l][1])-0.5*data[l][3],  (img.shape[0] - data[l][2])- 0.5* data[l][4]), data[l][3], data[l][4], linewidth=3, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        axs.add_patch(rect)

    plt.show()


class seed_context(object):
    """Creates a context with a custom random seed, then resets it.

    This allows for setting a custom seed (for reproducibility) in a
    context, then resetting the original state after exiting, to
    prevent interfering with the program execution.
    """

    def __init__(self, seed):
        self._seed = seed
        self._prev_state = None

    def __enter__(self):
        self._prev_state = np.random.get_state()
        np.random.seed(self._seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self._prev_state)
        self._prev_state = None

    def reset(self):
        np.random.seed(self._seed)

