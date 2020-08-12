import platform
import os
import time

# numpy/opencv
import numpy as np
import cv2 as cv

import pandas as pd

import bokeh
from bokeh.io import curdoc
from bokeh.events import SelectionGeometry
from bokeh.models import ColumnDataSource, HoverTool, Button, Div, BoxSelectTool, CustomJS
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer

# -----------------------------------------------------------------------------
# set up some global variables that will be used throughout the code
script_path = os.path.realpath(__file__)
image_path = os.path.dirname(os.path.dirname(script_path))

# data source to contain the input image and the crop selection parameters
p1_src = ColumnDataSource(data=dict(input_img=[], x=[], y=[], w=[], h=[]))




cb_dict = dict(p1_src=p1_src)
callback = CustomJS(args=cb_dict, code="""
    // get data source from Callback args
    var data = p1_src.data;

    /// get BoxSelectTool dimensions from cb_data parameter of Callback
    var geometry = cb_obj['geometry'];

    /// calculate Rect attributes
    var width = geometry['x1'] - geometry['x0'];
    var height = geometry['y1'] - geometry['y0'];
    var x = geometry['x0'];
    var y = geometry['y0'];

    console.log(x);
    console.log(y);
    console.log(width);
    console.log(height);
    
    /// update data source with new Rect attributes
    //data['x'].push(x);
    //data['y'].push(y);
    //data['w'].push(width);
    //data['h'].push(height);

    //p1_src.data = data;
    //p1_src.change.emit();
""")

#box_select = BoxSelectTool(callback=callback)

# Function definitions
# -----------------------------------------------------------------------------
def get_input():
    global detection_windows, results_div, filename_div, image_path

    image_name = "D:/Projects/dlib_object_detection/obj_det_lib/images/mframe_05042.png"


    print("Processing File: ", image_name)
    # load in an image
    image_path = os.path.dirname(image_name)
    color_img = cv.imread(image_name)

    # convert the image to RGBA for display
    rgba_img = cv.cvtColor(color_img, cv.COLOR_RGB2RGBA)
    p1_src.data = dict(input_img=[np.flipud(rgba_img)])





# Figure definitions
# -----------------------------------------------------------------------------
p1 = figure(x_range=(0,500), y_range=(0,350), plot_height=350, plot_width=500, title="Input image", tools=['pan', 'box_zoom', 'box_select', 'save', 'reset'], toolbar_location="right")
p1.image_rgba(image="input_img", x=0, y=0, dw=500, dh=350, source=p1_src)
p1.axis.visible = False
p1.grid.visible = False
#p1.x_range.range_padding = 0
#p1.y_range.range_padding = 0

get_input()

# -----------------------------------------------------------------------------
def selection_change(evt):
    geometry = evt.geometry

    x1 = min(geometry['x0'], geometry['x1'])
    y1 = min(geometry['y0'], geometry['y1'])

    x2 = max(geometry['x0'], geometry['x1'])
    y2 = max(geometry['y0'], geometry['y1'])

    print("test1")
    print(x1)
    print(x2)
    print(y1)
    print(y2)

#p1_src.selected.js_on_change('indices', callback)
#p1.js_on_event(SelectionGeometry, callback)

p1.on_event(SelectionGeometry, selection_change)

# Layout
# -----------------------------------------------------------------------------
layout = column(p1)

doc = curdoc()
doc.title = "Object Detection Viewer"
doc.add_root(layout)

#show(layout)
