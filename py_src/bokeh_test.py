import platform
import os
import math
import ctypes as ct
import numpy as np
import cv2 as cv
import bokeh
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Button
from bokeh.models.widgets import FileInput
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer

script_path = os.path.realpath(__file__)

# set up some global variables that will be used throughout the code
# read only
update_time = 200

use_webcam = False
if use_webcam:
    vc = cv.VideoCapture(0)
else:
    image_name = os.path.dirname(os.path.dirname(script_path)) + "/input_test.png"

print("image path: " + str(image_name))

# modify these to point to the right locations
if platform.system() == "Windows":
    libname = "mnist_lib.dll"
    lib_location = "D:/Projects/mnist_dll/build/Release/" + libname
    weights_file = "D:/Projects/mnist_dll/nets/mnist_net_pso_14_97.dat"
elif platform.system() == "Linux":
    libname = "libmnist_lib.so"
    home = os.path.expanduser('~')
    lib_location = home + "/Projects/mnist_net_lib/build/" + libname
    weights_file = home + "/Projects/mnist_net_lib/nets/mnist_net_pso_14_97.dat"
else:
    quit()


# read and write global



def button_callback(btn):
    global x
    bp = 1
    btn.label = "test2"

def update():
    global x

def upload_fit_data(attr, old, new):
    print("fit data upload succeeded")
    print(file_input.filename)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------



file_input = FileInput(accept=".csv,.json,.txt", title="File")
file_input.on_change('value', upload_fit_data)


b01 = Button(label='Test')
b01.on_click(lambda: button_callback(b01))

layout = column([b01, file_input])

# layout = column([row([column([p1, p2]), l12, l08]), row([Spacer(width=200, height=375), l02, l01])])

show(layout)

doc = curdoc()
doc.title = "MNIST Viewer"
doc.add_root(layout)
doc.add_periodic_callback(update, update_time)

# doc.hold('combine')
