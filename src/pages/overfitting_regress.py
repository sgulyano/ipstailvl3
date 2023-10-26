import dash
from dash import dcc
from dash import html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

dash.register_page(__name__, title="โมเดลการถดถอยที่เฉพาะเจาะจง/ง่ายเกินไป")

def get_layout():
    ## load data
    x = np.random.rand(100)*10 - 5
    y = 5/(1 + np.exp(-x)) + np.random.randn(100)

    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    idx = np.argsort(x_test)
    x_test = x_test[idx]
    y_test = y_test[idx]

    def get_data(x, degree=1):
        return [[x**(d+1) for d in range(degree)] for i in x]


    def get_fig(degree=1, show_dec_bound=True):
        x_deg_train = [[x**(d+1) for d in range(degree)] for x in x_train]
        x_deg_test = [[x**(d+1) for d in range(degree)] for x in x_test]
        reg = LinearRegression()
        reg.fit(x_deg_train, y_train)

        y_pred = reg.predict(x_deg_test)

        mse_tr = mean_squared_error(y_train, reg.predict(x_deg_train))
        mse_te = mean_squared_error(y_test, y_pred)

        fig = go.Figure(layout=go.Layout(
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title='X',
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title='Y',
        ))
        fig.add_trace(go.Scatter(x=x_train, y=y_train,
                                    mode='markers',
                                    name='Train',
                                    marker_color='black',
                                    marker_size=12,
                                    opacity=0.8))
        fig.add_trace(go.Scatter(x=x_test, y=y_test,
                                    mode='markers',
                                    name='Test',
                                    marker_color='gray',
                                    marker_size=12,
                                    opacity=0.8))
        fig.add_trace(go.Scatter(x=x_test, y=y_pred,
                                    mode='lines',
                                    name='Model',
                                    marker_color='red'))
        return fig, mse_tr, mse_te


    fig, mse_tr, mse_te = get_fig()

    degree_marks = {i: '' for i in range(1,17)}
    degree_marks[1] = '1'
    degree_marks[16] = '16'

    controls = dbc.Row([
        dbc.Row([
            html.H5(["Degree  ", dbc.Badge("4", className="ml-1", color="primary", id='overfit-regress-degree-label')]),
            dcc.Slider(
                id='overfit-regress-degree-slider-id',
                min=1,
                max=16,
                step=None,
                marks=degree_marks,
                value=1
            ),
        ]),
        html.Div([
            html.H5([" MSE "]),
            html.H6([" MSE บน training data =  ", dbc.Badge(f'{mse_tr:.3f}', className="ml-1", color="success", id='overfit-regress-mse-train-id')]),
            html.H6([" MSE บน test data = ", dbc.Badge(f'{mse_te:.3f}', className="ml-1", color="danger", id='overfit-regress-mse-test-id')]),
            ])
    ])

    ## Main layout
    layout = dbc.Container(
        [
            html.H1("โมเดลที่เฉพาะเจาะจงเกินไป VS โมเดลที่ง่ายเกินไป (Overfitting/Underfitting in Regression)"),
            html.Div(children='''
                ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่า hyperparamter degree ของ Linear Regression แล้วดูว่าเมื่อใดเกิด 
                overfitting/underfitting โดยการวาดกราฟเส้นระหว่างค่า MSE กับค่าตัวแปร Degree ของ Linear Regression 
                ของทั้ง training data และ test data
            '''),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(controls, md=4),
                    dbc.Col(dcc.Graph(id="overfit-regress-graph-id", figure=fig, animate=True), md=8),
                ],
                align="center",
            ),
        ],
        fluid=True,
    className="p-5")

    @callback(
        [Output(component_id='overfit-regress-degree-label', component_property='children'),
        Output(component_id='overfit-regress-graph-id', component_property='figure'),
        Output(component_id='overfit-regress-mse-train-id', component_property='children'),
        Output(component_id='overfit-regress-mse-test-id', component_property='children')],
        [Input(component_id='overfit-regress-degree-slider-id', component_property='value')]
    )
    def update_under_div(degree):
        fig, mse_tr, mse_te = get_fig(degree)
        return [f'{degree}', fig, f'{mse_tr:.3f}', f'{mse_te:.3f}']

    return layout

layout = get_layout()