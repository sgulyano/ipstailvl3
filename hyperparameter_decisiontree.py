import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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


    def get_fig(depth=4, max_leaf_nodes=10, min_impur_dec=0, show_dec_bound=True):
        clf = tree.DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impur_dec)
        clf = clf.fit(x_train, y_train)

        acc_tr = accuracy_score(y_train, clf.predict(x_train))
        acc_te = accuracy_score(y_test, clf.predict(x_test))

        fig = plt.figure(figsize=(9.6, 7.2))
        tree.plot_tree(clf)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        tree_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        tree_img = tree_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # print(tree_img.shape)
        

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

        colors = ['red', 'blue', 'green']
        for i, color in enumerate(colors):
            idx = np.where(y_train == i)
            fig.add_trace(go.Scatter(x=x_train[idx, 0].squeeze(), y=x_train[idx, 1].squeeze(),
                                    mode='markers',
                                    name=data.target_names[i],
                                    marker_color=color,
                                    opacity=0.8))
        for i, color in enumerate(colors):
            idx = np.where(y_test == i)
            fig.add_trace(go.Scatter(x=x_test[idx, 0].squeeze(), y=x_test[idx, 1].squeeze(),
                                    mode='markers',
                                    name=data.target_names[i] + ' test',
                                    marker_color=color,
                                    opacity=0.3))
        return fig, tree_img, acc_tr, acc_te


    fig, tree_img, acc_tr, acc_te = get_fig()


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
                max=16,
                marks={i: '{}'.format(i) for i in [1, 4, 7, 10, 13, 16]},
                value=4,
            ),
            dcc.Markdown('### Max Leaf Nodes'),
            dcc.Slider(
                id='leaf-slider-id',
                min=1,
                max=100,
                marks={i: '{}'.format(i) for i in range(1,10,100)},
                value=10,
            ),
            dcc.Markdown('### Min Impurity'),
            dcc.Slider(
                id='impur-slider-id',
                min=0,
                max=1,
                marks={i: '{}'.format(i) for i in np.arange(0,1,0.1)},
                value=0,
            ),
        ],
            style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}
        ),

        html.Div(children=[
            dcc.Graph(id='tree-id', 
                figure=go.Figure(go.Image(z=tree_img),
                    layout = go.Layout(
                        margin=go.layout.Margin(
                                l=0, #left margin
                                r=0, #right margin
                                b=0, #bottom margin
                                t=0, #top margin
                            ),
                        title = 'Overview',
                        xaxis = dict(showticklabels=False),
                        yaxis = dict(showticklabels=False)))),
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
        Output(component_id='accuracy-train-id', component_property='children'),
        Output(component_id='accuracy-test-id', component_property='children')],
        [Input(component_id='depth-slider-id', component_property='value'),
        Input(component_id='leaf-slider-id', component_property='value'),
        Input(component_id='impur-slider-id', component_property='value')]
    )
    def update_under_div(depth, max_leaf, min_impur):
        # print(depth, max_leaf, min_impur)
        fig, tree_img, acc_tr, acc_te = get_fig(depth, max_leaf, min_impur)
        return [fig, f'Train Accuracy = {acc_tr:.3f}', f'Test Accuracy = {acc_te:.3f}']

    return app

if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)



# import plotly.graph_objects as go
# # import plotly.tools as tls
# from skimage import data as data2

# from sklearn.datasets import load_iris
# from sklearn import tree
# import matplotlib.pyplot as plt
# import numpy as np

# clf = tree.DecisionTreeClassifier(random_state=0)
# iris = load_iris()

# clf = clf.fit(iris.data, iris.target)

# fig = plt.figure(figsize=(14.4, 10.8))
# tree.plot_tree(clf)  # doctest: +SKIP

# # If we haven't already shown or saved the plot, then we need to
# # draw the figure first...
# fig.canvas.draw()

# # Now we can save it to a numpy array.
# data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# print(data.shape)

# # img = data2.astronaut()s
# # img_rgb = [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
# #            [[0, 255, 0], [0, 0, 255], [255, 0, 0]]]
# fig = go.Figure(go.Image(z=data))
# fig.show()

