import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
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
            url_base_pathname='/imbalanceddata/',
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "จำนวนข้อมูลที่ไม่สมดุลกัน"

    ## load data
    data = load_breast_cancer()
    X = data['data'][:, :2]
    y = data['target']

    fn = ['รัศมีเฉลี่ย', 'ความขรุขระ']
    cn = ['มะเร็ง/เนื้อร้าย', 'เนื้องอก']

    idx = np.random.choice(np.where(y == 0)[0], size=int(
        np.sum(y == 1)*0.1), replace=False)

    noise = np.random.normal(0, 0.1, (idx.size*10, 2))
    noise[:,1] = noise[:,1] * 2
    noise[:idx.size, :] = 0

    x_train = np.concatenate((X[y == 1], X[idx]))
    y_train = np.concatenate((y[y == 1], y[idx]))

    plot_step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    def get_fig(x_train, y_train, show_dec_bound=False):
        clf = tree.DecisionTreeClassifier(
            random_state=0, max_depth=4, min_samples_split=10)
        clf = clf.fit(x_train, y_train)
        tree.plot_tree(clf)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_layout = go.Layout(
            uirevision=True,
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title=fn[0],
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title=fn[1],
        )
        if show_dec_bound:
            fig = go.Figure(data=go.Heatmap(
                z=Z, 
                x=np.arange(x_min, x_max, plot_step),
                y=np.arange(y_min, y_max, plot_step),
                colorscale=[[0, '#ef553b'], [1, '#636efa']],
                opacity=0.2,
                colorbar=dict(),
                showscale=False
            ), layout=fig_layout)
        else:
            fig = go.Figure(layout=fig_layout)

        colors = ['239, 85, 59', '99, 110, 250']
        symbols = ['circle', 'square']
        for i, (color, symbol) in enumerate(zip(colors, symbols)):
            idx = np.where(y_train == i)
            fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                     mode='markers',
                                     name=cn[i],
                                    #  marker_size=12,
                                     marker_color='rgb('+color+')',))
                                    #  marker_symbol=symbol,
                                    #  marker_line_color='rgb('+color+')', 
                                    #  marker_line_width=2))
        return fig

    fig = get_fig(x_train, y_train)

    ## Control
    under_spl_marks = {i: ''.format(i) for i in range(10, 101, 10)}
    under_spl_marks[10] = '10%'
    under_spl_marks[100] = '100%'
    over_spl_marks = {i: ''.format(i) for i in range(100, 1001, 100)}
    over_spl_marks[100] = '100%'
    over_spl_marks[1000] = '1000%'

    controls = dbc.Card([
        dbc.FormGroup([
            html.H5(["Under-Sampling", dbc.Badge("100%", className="ml-1", color="primary", id='under-spl-label')]),
            dcc.Slider(
                id='under-spl-slider-id',
                min=10,
                max=100,
                step=None,
                marks=under_spl_marks,
                value=100
            ),
        ]),
        dbc.FormGroup([
            html.H5(["Over-Sampling", dbc.Badge("100%", className="ml-1", color="primary", id='over-spl-label')]),
            dcc.Slider(
                id='over-spl-slider-id',
                dots=True,
                min=100,
                max=1000,
                step=None,
                marks=over_spl_marks,
                value=100,
            ),
        ]),
        dbc.FormGroup([
            # dbc.Label("Parameter"),
            html.H5(["Parameter"]),
            dbc.Checklist(
                options=[
                    {"label": "แสดงขอบเขตการจำแนก", "value": 1},
                ],
                value=[],
                id="show-dec-bound-id",
                inline=True,
                switch=True,
            ),
        ]),
    ],
        body=True,
    )

    ## Main layout
    app.layout = dbc.Container(
        [
            html.H1("จำนวนข้อมูลที่ไม่สมดุลกัน (Imbalanced Data)"),
            html.Div(children='''
                ในแบบฝึกหัดนี้ ให้ผู้เรียนลองใช้เทคนิค 1) การสุ่มข้อมูลจากกลุ่มหลักให้มีน้อยลง (Under-Sampling) และ 2) 
                การสร้างข้อมูลของกลุ่มย่อยให้มีมากขึ้น (Over-Sampling) แล้วลองสังเกต Decision Tree ผลลัพธ์ ที่ได้
            '''),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(controls, md=4),
                    dbc.Col(dcc.Graph(id="graph-id", figure=fig), md=8),
                ],
                align="center",
            ),
        ],
        fluid=True,
    )


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

    def update_under_div(under_ratio, over_ratio, show_decision_boundary):
        x_under, y_under = under_sampling(x_train, y_train, under_ratio)
        x_new, y_new = over_sampling(x_under, y_under, over_ratio)
        fig = get_fig(x_new, y_new, len(show_decision_boundary))
        return f'{under_ratio}%', f'{over_ratio}%', fig

    app.callback([Output('under-spl-label', 'children'),
                  Output('over-spl-label', 'children'),
                  Output('graph-id', 'figure')],
                 [Input("under-spl-slider-id", "value"),
                  Input('over-spl-slider-id', 'value'),
                  Input('show-dec-bound-id', 'value')])(
        update_under_div
    )

    return app


if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)
