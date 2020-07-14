import dash
import dash_core_components as dcc
import dash_html_components as html
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

def get_app(server=None):
    if server:
        app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname='/hyperparameters/'
        )
    else:
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


    ##
    data = load_iris()

    X = data['data'][:, :2]
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    fn = data.feature_names[:2]
    cn = data.target_names


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
                uniformtext_minsize=8,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[min(node['y'])-5,max(node['y'])+2], showgrid=False, zeroline=False, showticklabels=False))
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
        
        plot_step = 0.1
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                            np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        colorscale = [[0, 'peachpuff'], [0.5, 'lightcyan'], [1, 'lightgreen']]
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
                xaxis_title=fn[0],
                yaxis=dict(range=[y_min, y_max]),
                yaxis_title=fn[1],
            )
            )
        else:
            fig = go.Figure(layout=go.Layout(
                xaxis=dict(range=[x_min, x_max]),
                xaxis_title=fn[0],
                yaxis=dict(range=[y_min, y_max]),
                yaxis_title=fn[1],
            ))

        colors = ['red', 'blue', 'green']
        for i, color in enumerate(colors):
            idx = np.where(y_train == i)
            fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                    mode='markers',
                                    name=cn[i],
                                    marker_color=color,
                                    opacity=0.8))
        for i, color in enumerate(colors):
            idx = np.where(y_test == i)
            fig.add_trace(go.Scatter(x=x_test[idx, 0].squeeze(), y=x_test[idx, 1].squeeze(),
                                    mode='markers',
                                    name=cn[i] + ' test',
                                    marker_color=color,
                                    opacity=0.3))
        return fig, fig_tree, acc_tr, acc_te


    fig, fig_tree, acc_tr, acc_te = get_fig()


    app.layout = html.Div([
        html.H1(children='ไฮเปอร์พารามิเตอร์ของต้นไม้ตัดสินใจ'),

        html.Div(children='''
            ในแบบฝึกหัดนี้ ให้นักเรียนลองเปลี่ยนค่าไฮเปอร์พารามิเตอร์ (hyperparamter) ของต้นไม้ตัดสินใจ (Decision Tree) เช่น ความลึก ฯลฯ แล้วดูว่าเกิดอะไรขึ้นกับโมเดลที่ได้
        '''),
        html.Div(children=[
            dcc.Markdown('### Max Depth'),
            dcc.Slider(
                id='depth-slider-id',
                min=1,
                max=6,
                step=None,
                marks={i: '{}'.format(i) for i in range(1,7)},
                value=4,
            ),
            dcc.Markdown('### Max Leaf Nodes'),
            dcc.Slider(
                id='leaf-slider-id',
                min=3,
                max=18,
                step=None,
                marks={i: '{}'.format(i) for i in range(3,19,3)},
                value=18,
            ),
            dcc.Markdown('### Min Samples Split'),
            dcc.Slider(
                id='impur-slider-id',
                min=2,
                max=40,
                step=None,
                marks={i: '{}'.format(i) for i in [2,5,10,20,40]},
                value=2,
            ),
        ],
            style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),

        html.Div(children=[
            dcc.Graph(id='decision-tree-id', figure=fig_tree),
            dcc.Graph(id='graph-id', figure=fig),
            html.Div([
                html.Div(id='accuracy-train-id', children=f'Train Accuracy = {acc_tr:.3f}'),
                html.Div(id='accuracy-test-id', children=f'Test Accuracy = {acc_te:.3f}')
            ],
                style={'textAlign': 'center'}
            )
        ],
            style={'width': '50%', 'display': 'inline-block'}
        )
    ])


    @app.callback(
        [Output(component_id='graph-id', component_property='figure'),
        Output(component_id='decision-tree-id', component_property='figure'),
        Output(component_id='accuracy-train-id', component_property='children'),
        Output(component_id='accuracy-test-id', component_property='children')],
        [Input(component_id='depth-slider-id', component_property='value'),
        Input(component_id='leaf-slider-id', component_property='value'),
        Input(component_id='impur-slider-id', component_property='value')]
    )
    def update_under_div(depth, max_leaf, min_impur):
        fig, fig_tree, acc_tr, acc_te = get_fig(depth, max_leaf, min_impur)
        return [fig, fig_tree,  f'Train Accuracy = {acc_tr:.3f}', f'Test Accuracy = {acc_te:.3f}']

    return app

if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)