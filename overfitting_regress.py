import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

def get_app(server=None):
    if server:
        app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname='/overfitting_regress/'
        )
    else:
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    ##
    data = load_boston()

    X = data['data'][:, 9]
    y = data['target']


    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    idx = np.argsort(x_train)
    x_train = x_train[idx]
    y_train = y_train[idx]

    x_min, x_max = X.min() - 1, X.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    train_mse = []
    test_mse = []
    for degree in range(1,16):
        x_deg_train = [[x**(d+1) for d in range(degree)] for x in x_train]
        x_deg_test = [[x**(d+1) for d in range(degree)] for x in x_test]

        reg = LinearRegression()
        reg.fit(x_deg_train, y_train)

        mse_tr = mean_squared_error(y_train, reg.predict(x_deg_train))
        mse_te = mean_squared_error(y_test, reg.predict(x_deg_test))

        train_mse.append(mse_tr)
        test_mse.append(mse_te)


    def get_data(x, degree=1):
        return [[x**(d+1) for d in range(degree)] for i in x]



    def get_fig(degree=1, show_dec_bound=True):
        x_deg_train = [[x**(d+1) for d in range(degree)] for x in x_train]
        x_deg_test = [[x**(d+1) for d in range(degree)] for x in x_test]
        reg = LinearRegression()
        reg.fit(x_deg_train, y_train)

        y_pred = reg.predict(x_deg_train)

        mse_tr = mean_squared_error(y_train, y_pred)
        mse_te = mean_squared_error(y_test, reg.predict(x_deg_test))
    #     clf = tree.DecisionTreeClassifier(max_depth=depth)
    #     clf = clf.fit(x_train, y_train)

    #     acc_tr = accuracy_score(y_train, clf.predict(x_train))
    #     acc_te = accuracy_score(y_test, clf.predict(x_test))

    #     tree.plot_tree(clf)

    #     plot_step = 0.1
    #     x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    #     y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                          np.arange(y_min, y_max, plot_step))

    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)

    #     colorscale = [[0, 'peachpuff'], [1, 'lightcyan']]

        fig = go.Figure(layout=go.Layout(
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title=data.feature_names[9],
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title='disease progression',
        ))
        fig.add_trace(go.Scatter(x=x_train, y=y_train,
                                    mode='markers',
                                    name='Train',
                                    marker_color='black',
                                    opacity=0.8))
        fig.add_trace(go.Scatter(x=x_test, y=y_test,
                                    mode='markers',
                                    name='Test',
                                    marker_color='gray',
                                    opacity=0.8))
        fig.add_trace(go.Scatter(x=x_train, y=y_pred,
                                    mode='lines',
                                    name='Model',
                                    marker_color='red'))

    #     colors = ['red', 'blue']
    #     for i, color in enumerate(colors):
    #         idx = np.where(y_train == i)
    #         fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
    #                                  mode='markers',
    #                                  name=data.target_names[i],
    #                                  marker_color=color,
    #                                  opacity=0.8))
    #     for i, color in enumerate(colors):
    #         idx = np.where(y_test == i)
    #         fig.add_trace(go.Scatter(x=x_test[idx, 0].squeeze(), y=x_test[idx, 1].squeeze(),
    #                                  mode='markers',
    #                                  name=data.target_names[i] + ' test',
    #                                  marker_color=color,
    #                                  opacity=0.3))
        return fig, mse_tr, mse_te


    fig, mse_tr, mse_te = get_fig()


    app.layout = html.Div([
        html.H1(children='Overfitting/Underfitting in Regression'),

        html.Div(children='''
            ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่า hyperparamter degree ของ Linear Regression แล้วดูว่าเมื่อใดเกิด overfitting/underfitting
        '''),
        html.Div(children=[
            # dcc.Markdown('### ชุดข้อมูล'),
            # dcc.Dropdown(
            #     options=[
            #         {'label': 'มะเร็งเต้านม', 'value': 'breast_cancer'},
            #     ],
            #     value='breast_cancer'
            # ),

            dcc.Markdown('### Degree'),
            dcc.Slider(
                id='degree-slider-id',
                min=1,
                max=16,
                marks={i: '{}'.format(i) for i in [1, 4, 7, 10, 13, 16]},
                value=1,
            ),

            dcc.Graph(figure=go.Figure([go.Scatter(x=list(range(1,16)), y=train_mse, mode='lines+markers', name="Train"), 
                                go.Scatter(x=list(range(1,16)), y=test_mse, mode='lines+markers', name="Test")],
                                layout=go.Layout(
                title='ข้อมูลที่ใช้ Train โมเดล',
                # xaxis=dict(range=[0, 17]),
                xaxis_title='Degree',
                # yaxis=dict(range=[0, 1.1]),
                yaxis_title='MSE',
            )))
        ],
            style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),

        html.Div(children=[
            dcc.Graph(id='graph-id', figure=fig),
            html.Div([
                html.Div(id='mse-train-id', children=f'Train MSE = {mse_tr:.3f}'),
                html.Div(id='mse-test-id', children=f'Test MSE = {mse_te:.3f}')
            ],
                style={'textAlign': 'center'}
            )
        ],
            style={'width': '50%', 'display': 'inline-block'}
        )
    ])


    @app.callback(
        [Output(component_id='graph-id', component_property='figure'),
        Output(component_id='mse-train-id', component_property='children'),
        Output(component_id='mse-test-id', component_property='children')],
        [Input(component_id='degree-slider-id', component_property='value')]
    )
    def update_under_div(degree):
        fig, mse_tr, mse_te = get_fig(degree)
        return [fig, f'Train MSE = {mse_tr:.3f}', f'Test MSE = {mse_te:.3f}']


if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)
