import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Slider, RadioButtonGroup, Div, Paragraph
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark
from sklearn import tree
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

## load data
data = load_breast_cancer()
X = data['data'][:, :2]
y = data['target']

# get min/max data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# make group 0 #sample 10% of group 1
idx = np.random.choice(np.where(y == 0)[0], size=int(
    np.sum(y == 1)*0.1), replace=False)

# pre-compute noise
noise = np.random.normal(0, 0.1, (idx.size*10, 2))
noise[:idx.size, :] = 0

# data for demonstration
x_train = np.concatenate((X[y == 1], X[idx]))
y_train = np.concatenate((y[y == 1], y[idx]))


y_set = ['มะเร็ง/เนื้อร้าย', 'เนื้องอก']
y_map_value = {k:v for k, v in enumerate(y_set)}
y_train_name = [y_map_value[y] for y in y_train]


plot_step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

def getZ(x_data, y_data):
    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4, min_samples_split=10)
    clf = clf.fit(x_data, y_data)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    return Z.reshape(xx.shape)

Z = getZ(x_train, y_train)

# Set up data
source = ColumnDataSource(data=dict(x=x_train[:,0], y=x_train[:,1], c=y_train_name))
bound_source = ColumnDataSource({'value': [Z]})

# Set up plot
plot = figure(plot_height=400, plot_width=600, title='ข้อมูลผู้ป่วยมะเร็งเต้านม',
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[x_min, x_max], y_range=[y_min, y_max],
              x_axis_label='รัศมีเฉลี่ย',
              y_axis_label='ความขรุขระ')

img = plot.image('value', source=bound_source, x=x_min, y=y_min, dw=x_max-x_min, dh=y_max-y_min, 
                 visible=False,
                 palette=('#d2e2ff', '#ffe0cc'))

plot.scatter('x', 'y', source=source, legend_field="c", fill_alpha=0.4, size=12, 
             marker=factor_mark('c', ['hex', 'triangle'], y_set),
             color=factor_cmap('c', 'Category10_3', y_set))

# Set up dashboard
title = Div(text="""<H1>จำนวนข้อมูลที่ไม่สมดุลกัน (Imbalanced Data)</H1>""")
desc = Paragraph(text="""ในแบบฝึกหัดนี้ ให้นักเรียนลองใช้เทคนิค 1) การสุ่มข้อมูลจากกลุ่มหลักให้มีน้อยลง (Under-Sampling) และ 2) 
การสร้างข้อมูลของกลุ่มย่อยให้มีมากขึ้น (Over-Sampling) แล้วลองสังเกต Decision Tree ผลลัพธ์ ที่ได้""")
header = column(title, desc, sizing_mode="scale_both")

# Set up widgets
text = Div(text="<H3>ตัวแปร</H3>")
underspl = Slider(title="Under-Sampling", value=100, start=10, end=100, step=10)
overspl = Slider(title="Over-Sampling", value=100, start=100, end=1000, step=100)


def under_sampling(x_train, y_train, ratio):
    num = int(np.sum(y_train == 1)*ratio/100.0)
    idx = np.where(y_train == 1)[0][:num]

    x_new = np.concatenate((x_train[y_train == 0], x_train[idx]))
    y_new = np.concatenate((y_train[y_train == 0], y_train[idx]))
    return x_new, y_new

def over_sampling(x_train, y_train, ratio, noise=noise):
    n = np.sum(y_train == 0)
    num = int(n*ratio/100.0)
    pos = np.arange(num) % n

    idx = np.where(y_train == 0)[0][pos]
    x_new = np.concatenate(
        (x_train[idx]+noise[:pos.shape[0]], x_train[y_train == 1]))
    y_new = np.concatenate((y_train[idx], y_train[y_train == 1]))
    return x_new, y_new

def update_data(attrname, old, new):
    # Get the current slider values
    under_ratio = underspl.value
    over_ratio = overspl.value

    x_under, y_under = under_sampling(x_train, y_train, under_ratio)
    x_new, y_new = over_sampling(x_under, y_under, over_ratio)

    y_train_name = [y_map_value[y] for y in y_new]

    source.data = dict(x=x_new[:,0], y=x_new[:,1], c=y_train_name)

    Z = getZ(x_new, y_new)
    bound_source.data = {'value': [Z]}
    

for w in [underspl, overspl]:
    w.on_change('value', update_data)


decbound_text = Paragraph(text="ขอบเขตการจำแนก")
decbound = RadioButtonGroup(labels=["ซ่อน", "แสดง"], active=0)

def update_img(attr, old, new):
    img.visible = bool(decbound.active)

decbound.on_change('active', update_img)


# Set up layouts and add to document
inputs = column(text, underspl, overspl, decbound_text, decbound)
body = row(inputs, plot, width=800)

curdoc().add_root(column(header, body))
curdoc().title = "จำนวนข้อมูลที่ไม่สมดุลกัน"