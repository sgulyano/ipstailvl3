import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

def get_app(server=None):
    if server:
        app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname='/imbalanceddata/'
        )
    else:
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


    ##
    data = load_breast_cancer()
    X = data['data'][:, :2]
    y = data['target']

    idx = np.random.choice(np.where(y == 0)[0], size=int(
        np.sum(y == 1)*0.1), replace=False)

    noise = np.random.normal(0, 0.1, (idx.size*10, 2))
    noise[:idx.size, :] = 0

    x_train = np.concatenate((X[y == 1], X[idx]))
    y_train = np.concatenate((y[y == 1], y[idx]))


    def get_fig(x_train, y_train, show_dec_bound=False):
        clf = tree.DecisionTreeClassifier(
            random_state=0, max_depth=4, min_samples_split=10)
        clf = clf.fit(x_train, y_train)
        tree.plot_tree(clf)

        plot_step = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                            np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        colorscale = [[0, 'peachpuff'], [1, 'lightcyan']]
        if show_dec_bound:
            fig = go.Figure(data=go.Heatmap(
                z=Z,
                x=np.arange(x_min, x_max, plot_step),
                y=np.arange(y_min, y_max, plot_step),
                colorscale=colorscale,
                colorbar=dict(),
                showscale=False
                # colorbar=dict(nticks=10, ticks='outside',
                #               ticklen=5, tickwidth=1,
                #               showticklabels=True,
                #               tickangle=0, tickfont_size=12)
            ),
                layout=go.Layout(
                xaxis=dict(range=[x_min, x_max]),
                xaxis_title=data.feature_names[0],
                yaxis=dict(range=[y_min, y_max]),
                yaxis_title=data.feature_names[1],
            )
            )
        else:
            fig = go.Figure(layout=go.Layout(
                xaxis=dict(range=[x_min, x_max]),
                xaxis_title=data.feature_names[0],
                yaxis=dict(range=[y_min, y_max]),
                yaxis_title=data.feature_names[1],
            ))

        colors = ['red', 'blue']
        for i, color in enumerate(colors):
            idx = np.where(y_train == i)
            fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                    mode='markers',
                                    name=data.target_names[i],
                                    marker_color=color))
        return fig


    fig = get_fig(x_train, y_train)


    app.layout = html.Div([
        html.H1(children='จำนวนข้อมูลที่ไม่สมดุลกัน (Imbalanced Data)'),

        html.Div(children='''
            ในแบบฝึกหัดนี้ ให้นักเรียนลองใช้เทคนิค 1) การสุ่มข้อมูลจากกลุ่มหลักให้มีน้อยลง (Under-Sampling) และ 2) 
    การสร้างข้อมูลของกลุ่มย่อยให้มีมากขึ้น (Over-Sampling) แล้วลองสังเกต Decision Tree ผลลัพธ์ ที่ได้
        '''),
        html.Div(children=[
            # dcc.Markdown('### ชุดข้อมูล'),
            # dcc.Dropdown(
            #     options=[
            #         {'label': 'มะเร็งเต้านม', 'value': 'breast_cancer'},
            #     ],
            #     value='breast_cancer'
            # ),

            dcc.Markdown('### Under-Sampling'),
            dcc.Slider(
                id='under-spl-slider-id',
                min=10,
                max=100,
                marks={i: '{}%'.format(i) for i in range(10, 101, 10)},
                value=100,
            ),

            dcc.Markdown('### Over-Sampling'),
            dcc.Slider(
                id='over-spl-slider-id',
                min=100,
                max=1000,
                marks={i: '{}%'.format(i) for i in range(100, 1001, 100)},
                value=100,
            ),

            dcc.Markdown('### Parameter'),
            dcc.Checklist(
                id='show-dec-bound-id',
                options=[
                    {'label': 'แสดง Decision Boundary',
                        'value': 'show_decision_boundary'},
                ],
                value=[]
            )
        ],
            style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),

        html.Div(children=[
            dcc.Graph(id='graph-id', figure=fig),
        ],
            style={'width': '50%', 'display': 'inline-block'}
        )
    ])


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


    @app.callback(
        Output(component_id='graph-id', component_property='figure'),
        [Input(component_id='under-spl-slider-id', component_property='value'),
        Input(component_id='over-spl-slider-id', component_property='value'),
        Input(component_id='show-dec-bound-id', component_property='value')]
    )
    def update_under_div(under_ratio, over_ratio, show_decision_boundary):
        x_under, y_under = under_sampling(x_train, y_train, under_ratio)
        x_new, y_new = over_sampling(x_under, y_under, over_ratio)
        fig = get_fig(x_new, y_new, len(show_decision_boundary))
        return fig
    
    return app


if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)
