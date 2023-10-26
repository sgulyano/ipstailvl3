import dash
from dash import dcc
from dash import html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

import myplotly_tree

np.random.seed(42)

dash.register_page(__name__, title="ไฮเปอร์พารามิเตอร์ของต้นไม้ตัดสินใจ")

def get_layout():
    data = load_iris()

    X = data['data'][:, :2]
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    fn = data.feature_names[:2]
    cn = data.target_names
    # print(cn)

    plot_step = 0.1
    x_min, x_max = X[:, 0].min() - .3, X[:, 0].max() + .3
    y_min, y_max = X[:, 1].min() - .3, X[:, 1].max() + .3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))

    def get_fig(depth=4, max_leaf_nodes=10, min_impur_dec=2, show_dec_bound=True):
        clf = tree.DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_impur_dec)
        clf = clf.fit(x_train, y_train)

        node, edge, anno = myplotly_tree.get_node_edge(clf, fn, cn, px_init = 10, py_init = 10)

        node_trace = go.Scatter(
            x=node['x'], y=node['y'],
            mode='markers+text',
            hoverinfo='text',
            text=node['text'],
            textposition="bottom center",
            marker=dict(
                showscale=False,
                color=node['color'],
                size=30,
                line_width=2))

        edge_trace = go.Scatter(
            x=edge['x'], y=edge['y'],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        fig_tree = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='ต้นไม้ตัดสินใจ',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[min(node['y'])-20,max(node['y'])+5], showgrid=False, zeroline=False, showticklabels=False))
                )
        for x, y, t in zip(anno['x'], anno['y'], anno['text']):
            fig_tree.add_annotation(
                        x=x,
                        y=y,
                        text=t,
                        showarrow=False,
                        font=dict(family="Courier New, monospace", size=16))


        acc_tr = accuracy_score(y_train, clf.predict(x_train))
        acc_te = accuracy_score(y_test, clf.predict(x_test))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=np.arange(x_min, x_max, plot_step),
            y=np.arange(y_min, y_max, plot_step),
            colorscale=[[0, '#e58139'], [0.5,'#39e581'], [1, '#a775ed']],
            opacity=0.2,
            colorbar=dict(),
            showscale=False
            # colorbar=dict(nticks=10, ticks='outside',
            #               ticklen=5, tickwidth=1,
            #               showticklabels=True,
            #               tickangle=0, tickfont_size=12)
        ), layout=go.Layout(
            uirevision=True,
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title=fn[0],
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title=fn[1],
        ))

        colors = ['229, 129, 57', '57, 229, 129', '167, 117, 237']
        for i, color in enumerate(colors):
            idx = np.where(y_train == i)
            fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                    mode='markers',
                                    name=cn[i],
                                    marker_size=10,
                                    marker_color='rgb('+color+')'))
        for i, color in enumerate(colors):
            idx = np.where(y_test == i)
            fig.add_trace(go.Scatter(x=x_test[idx, 0].squeeze(), y=x_test[idx, 1].squeeze(),
                                    mode='markers',
                                    name=cn[i] + ' test',
                                    marker_line_color='rgba('+color+', 1.0)',
                                    marker_size=10,
                                    marker_color='rgba('+color+', 0.5)',
                                    marker_line_width=1))
        return fig, fig_tree, acc_tr, acc_te


    fig, fig_tree, acc_tr, acc_te = get_fig()

    depth_marks = {i: '' for i in range(1, 7)}
    depth_marks[1] = '1'
    depth_marks[6] = '6'

    leaf_marks = {i: '' for i in range(3,19,3)}
    leaf_marks[3] = '3'
    leaf_marks[18] = '18'

    controls = dbc.Row([
        dbc.Row([
            html.H5(["Max Depth  ", dbc.Badge(
                "4", className="ml-1", color="primary", id='depth-label')]),
            dcc.Slider(
                id='depth-slider-id',
                min=1,
                max=6,
                step=None,
                marks=depth_marks,
                value=4
            ),
        ]),
        dbc.Row([
            html.H5(["Max Leaf Nodes  ", dbc.Badge(
                "18", className="ml-1", color="primary", id='leaf-label')]),
            dcc.Slider(
                id='leaf-slider-id',
                min=3,
                max=18,
                step=None,
                marks=leaf_marks,
                value=18
            ),
        ]),
        dbc.Row([
            html.H5(["Min Samples Split  ", dbc.Badge(
                "2", className="ml-1", color="primary", id='spl-label')]),
            dcc.Slider(
                id='spl-slider-id',
                min=2,
                max=40,
                step=None,
                marks={i: '{}'.format(i) for i in [2,5,10,20,40]},
                value=2
            ),
        ]),

        html.Div([
            html.H5([" ความแม่นยำ "]),
            html.H6([" ความแม่นยำ บน training data =  ", dbc.Badge(
                f'{acc_tr:.3f}', className="ml-1", color="success", id='accuracy-train-id')]),
            html.H6([" ความแม่นยำ บน test data = ", dbc.Badge(
                f'{acc_te:.3f}', className="ml-1", color="danger", id='accuracy-test-id')]),
        ])
    ])

    ## Main layout
    layout = dbc.Container(
        [
            html.H1("ไฮเปอร์พารามิเตอร์ของต้นไม้ตัดสินใจ (Hyperparameter)"),
            html.Div(children='''
                ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่าไฮเปอร์พารามิเตอร์ (hyperparamter) ของต้นไม้ตัดสินใจ 
                (Decision Tree) เช่น ความลึก ฯลฯ แล้วดูว่าเกิดอะไรขึ้นกับโมเดลที่ได้
            '''),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(controls, md=4, align='start'),
                    dbc.Col([
                        dcc.Graph(id='decision-tree-id', figure=fig_tree, animate=True),
                        dcc.Graph(id="graph-id", figure=fig)]
                        , md=8),
                ],
                align="center",
            ),
        ],
        fluid=True,
    className="p-5")


    @callback(
        [Output(component_id='graph-id', component_property='figure'),
        Output(component_id='decision-tree-id', component_property='figure'),
        Output(component_id='accuracy-train-id', component_property='children'),
        Output(component_id='accuracy-test-id', component_property='children'),
        Output(component_id='depth-label', component_property='children'),
        Output(component_id='leaf-label', component_property='children'),
        Output(component_id='spl-label', component_property='children')],
        [Input(component_id='depth-slider-id', component_property='value'),
        Input(component_id='leaf-slider-id', component_property='value'),
        Input(component_id='spl-slider-id', component_property='value')]
    )
    def update_under_div(depth, max_leaf, min_spl):
        fig, fig_tree, acc_tr, acc_te = get_fig(depth, max_leaf, min_spl)
        return [fig, fig_tree,  f'{acc_tr:.3f}', f'{acc_te:.3f}', f'{depth}', f'{max_leaf}', f'{min_spl}']

    return layout

layout = get_layout()