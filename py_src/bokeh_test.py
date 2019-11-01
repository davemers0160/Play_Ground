import platform
import os
import math
import ctypes as ct
import sys

import numpy as np
import cv2 as cv
# import tkinter as tk
# from tkinter import filedialog
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication
import bokeh
from bokeh import events
from bokeh.io import curdoc, output_file, show
from bokeh.layouts import widgetbox
from bokeh.models import ColumnDataSource, Button, CustomJS
from bokeh.models.widgets import FileInput, TextInput, Paragraph, PreText, Div, TableColumn, DataTable, DateFormatter
from bokeh.plotting import figure  #, show
from bokeh.layouts import column, row, Spacer

script_path = os.path.realpath(__file__)

# set up some global variables that will be used throughout the code
# read only
update_time = 200
app = QApplication([""])
div = Div(width=1000)
index = 0

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


def button_callback():
    global index, div
    file_path = QFileDialog.getOpenFileName(None, "Select a file",  os.path.dirname(os.path.dirname(script_path)), "Image files (*.png *.jpg *.gif);;All files (*.*)")
    # print(file_path[0])

    results = "<font size='3'>"
    for idx in range(4):
        results += str(idx) + ", " + str(idx) + ", " + str(idx) + ", " + str(idx) + ", " + "label" + "<br>"
    # results += str(index) + ": " + file_path[0] + "<br>"
    # results += str(index) + ": " + file_path[0] + "<br>"
    results += file_path[0] + "<br>"
    results += "</font>"
    div.text = results #"<font size='4'>" + str(index) + ": " + file_path[0] + "</font>"


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

button = Button(label='Select File', width=100)
button.on_click(button_callback)

layout = widgetbox(button, div)

show(layout)

doc = curdoc()
doc.title = "MNIST Viewer"
doc.add_root(layout)
#doc.add_periodic_callback(update, update_time)

# doc.hold('combine')
