from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider, Spinner
from bokeh.plotting import Figure, output_file, show

output_file("js_on_change.html")

limits = [0, 5000]
l2 = [limits, limits, limits]
step = 1000

x = [x*0.005 for x in range(0, 200)]
y = x

source = ColumnDataSource(data=dict(x=x, y=y))
f = Spinner(title="focal length (mm)", low=0.1, high=500, step=0.1, value=55, width=100)
plot = Figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

cb_dict = dict(source=source, plot=plot, fs=f, limits=limits, step=step, l2=l2)
callback = CustomJS(args=cb_dict, code="""
    var data = source.data;
    var f = cb_obj.value;
    var x = data['x'];
    var y = data['y'];
    var r= [];
    
    var start = Math.max(limits[0], step)   
    var num = Math.floor((limits[1] - start) / step);
    
    console.log(fs.value);
    console.log(limits[1]);
    console.log(start);
    console.log(l2[0]);
    //console.log(num);
    
    for(var idx = 0; idx<num; idx++)
    {
        r.push(((idx+1)*step)/1000);
    }
    
    console.log(r)
    
    for (var i = 0; i < x.length; i++) {
        y[i] = Math.pow(x[i], f);
    }
    source.change.emit();
    plot.x_range.end = 10;
""")

slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")
slider.js_on_change('value', callback)

layout = column(slider, plot, f)

show(layout)