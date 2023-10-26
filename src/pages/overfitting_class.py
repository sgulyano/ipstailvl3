import dash
from dash import dcc
from dash import html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

dash.register_page(__name__, title="โมเดลสำหรับจำแนกที่เฉพาะเจาะจง-ง่ายเกินไป")

def get_layout():
    data = load_breast_cancer()

    X = data['data'][:, :2]
    y = data['target']

    fn = ['รัศมีเฉลี่ย', 'ความขรุขระ']
    cn = ['มะเร็ง/เนื้อร้าย', 'เนื้องอก']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    plot_step = 0.1
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))


    def get_fig(depth=4):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(x_train, y_train)

        acc_tr = accuracy_score(y_train, clf.predict(x_train))
        acc_te = accuracy_score(y_test, clf.predict(x_test))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=np.arange(x_min, x_max, plot_step),
            y=np.arange(y_min, y_max, plot_step),
            colorscale=[[0, '#ef553b'], [1, '#636efa']],
            opacity=0.2,
            colorbar=dict(),
            showscale=False

        ), layout=go.Layout(
            uirevision=True,
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title=fn[0],
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title=fn[1],
        ))

        colors = ['239, 85, 59', '99, 110, 250']
        symbols = ['circle', 'square']
        for i, (color, symbol) in enumerate(zip(colors, symbols)):
            idx = np.where(y_train == i)
            fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                     mode='markers',
                                     name=cn[i] + ' train',
                                     marker_color='rgb('+color+')',))

        for i, color in enumerate(colors):
            idx = np.where(y_test == i)
            fig.add_trace(go.Scatter(x=x_test[idx, 0].squeeze(), y=x_test[idx, 1].squeeze(),
                                     mode='markers',
                                     name=cn[i] + ' test',
                                     marker_color='rgba('+color+', 0.4)',
                                     marker_line_color='rgba('+color+', 1.0)',
                                     marker_line_width=1))

        return fig, acc_tr, acc_te

    fig, acc_tr, acc_te = get_fig()

    depth_marks = {i: '' for i in range(1, 11)}
    depth_marks[1] = '1'
    depth_marks[10] = '10'

    controls = dbc.Row([
        dbc.Row([
            html.H5(["Max Depth  ", dbc.Badge(
                "4", className="ml-1", color="primary", id='overfit-class-depth-label')]),
            dcc.Slider(
                id='overfit-class-depth-slider-id',
                min=1,
                max=10,
                step=None,
                marks=depth_marks,
                value=4
            ),
        ]),
        html.Div([
            html.H5([" ความแม่นยำ "]),
            html.H6([" ความแม่นยำ บน training data =  ", dbc.Badge(
                f'{acc_tr:.3f}', className="ml-1", color="success", id='overfit-class-accuracy-train-id')]),
            html.H6([" ความแม่นยำ บน test data = ", dbc.Badge(
                f'{acc_te:.3f}', className="ml-1", color="danger", id='overfit-class-accuracy-test-id')]),
        ])
    ])

    ## Main layout
    layout = dbc.Container(
        [
            html.H1(
                "โมเดลที่เฉพาะเจาะจงเกินไป VS โมเดลที่ง่ายเกินไป (Overfitting/Underfitting in Classification)"),
            html.Div(children='''
                ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่าตัวแปร depth ของ Decision Tree แล้วดูว่าเมื่อใดก่อให้เกิดโมเดลที่เฉพาะเจาะจงเกินไป 
                (overfitting) และ โมเดลที่ง่ายเกินไป (underfitting) โดยการวาดกราฟเส้นระหว่างความแม่นยำกับค่าตัวแปร depth 
                ของ Decision Tree ของทั้ง training data และ test data
            '''),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(controls, md=4),
                    dbc.Col(dcc.Graph(id="overfit-class-graph-id", figure=fig), md=8),
                ],
                align="center",
            ),
        ],
        fluid=True,
    className="p-5")

    @callback(
        [Output(component_id='overfit-class-depth-label', component_property='children'),
         Output(component_id='overfit-class-graph-id', component_property='figure'),
         Output(component_id='overfit-class-accuracy-train-id',
                component_property='children'),
         Output(component_id='overfit-class-accuracy-test-id', component_property='children')],
        [Input(component_id='overfit-class-depth-slider-id', component_property='value')]
    )
    def update_under_div(depth):
        fig, acc_tr, acc_te = get_fig(depth)
        return [f'{depth}', fig, f'{acc_tr:.3f}', f'{acc_te:.3f}']
    
    return layout

layout = get_layout()