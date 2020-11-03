import numpy as np
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_app(server=None):
    if server:
        app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname='/samplingbias/',
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "อคติในการเลือกตัวอย่าง"

    x = np.random.rand(100)*10 - 5
    y = 5/(1 + np.exp(-x)) + np.random.randn(100)
    # x = np.random.rand(100)
    # y = np.random.rand(100)

    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1

    reg = LinearRegression().fit(x.reshape(x.size,1), y)
    y_pred_best = reg.predict([[x_min], [x_max]])

    mse_best = mean_squared_error(y, reg.predict(x.reshape(x.size,1)))


    f_input = go.Figure([go.Scatter(x=x, y=y, mode='markers')],
                        layout=go.Layout(
                            title='ข้อมูลทั้งหมด',
                            margin=dict(b=0, l=20, r=20, t=40),
                            xaxis=dict(range=[x_min, x_max]),
                            xaxis_title='X',
                            yaxis=dict(range=[y_min, y_max]),
                            yaxis_title='Y',
    ))
    f_input.layout.dragmode = 'lasso'


    app.layout = dbc.Container(
        [
            html.H1("อคติในการเลือกตัวอย่าง (Sampling Bias)"),
            html.Div(children='''
                ในกราฟทางซ้าย ให้ผู้เรียนลองเลือกจุดมาบางส่วน แล้วดูว่าผลลัพธ์ Linear Regression ที่ได้ ต่างจาก ถ้าใช้ข้อมูลทั้งหมดที่มีมากน้อยแค่ไหน
            '''),
            html.Hr(),
            dbc.Row([
                    dbc.Col(dcc.Graph(id='input-graph', figure=f_input), md=6),
                    dbc.Col(dcc.Graph(id='output-graph'), md=6)
                ],
                align="center",
            ),
            html.Div([
                html.H5([" MSE ของ โมเดลจากข้อมูลทั้งหมด ", dbc.Badge(f'{mse_best:.3f}', className="ml-1", color="success", id='accuracy-best-id')]),
                html.H5([" MSE ของ โมเดลที่ได้ ", dbc.Badge("inf", className="ml-1", color="danger", id='accuracy-id')]),
                # html.Div(id='accuracy-best-id', children=f'MSE = {mse_best:.3f}'),
                # html.Div(id='accuracy-id')
            ],
            style={'textAlign': 'center'}
            )
        ],
        fluid=True,
    )

    @app.callback([Output(component_id='output-graph', component_property='figure'),
                Output(component_id='accuracy-id', component_property='children')], 
                [Input('input-graph', 'selectedData')])
    def display_selected_data(selectedData):
        output_layout = go.Layout(
            title='ข้อมูลที่ใช้ Train โมเดล',
            margin=dict(b=0, l=20, r=20, t=40),
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title='X',
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title='Y',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        if selectedData and selectedData["points"]:
            idx = [row['pointIndex'] for row in selectedData["points"]]

            reg = LinearRegression().fit(x[idx].reshape(len(idx),1), y[idx])
            mse = mean_squared_error(y, reg.predict(x.reshape(x.size,1)))

            y_pred = reg.predict([[x_min], [x_max]])

            f_output = go.Figure([go.Scatter(x=x[idx], y=y[idx], mode='markers', name="Selected Data", showlegend=False), 
                                go.Scatter(x=[x_min, x_max], y=y_pred, mode='lines', name="โมเดลที่ได้"),
                                go.Scatter(x=[x_min, x_max], y=y_pred_best, mode='lines', name="โมเดลจากข้อมูลทั้งหมด")],
                                layout=output_layout)
        else:
            f_output = go.Figure(layout=output_layout)
            mse = np.inf
        # print(json.dumps(selectedData, indent=2))
        return [f_output, f'{mse:.3f}']

    return app

if __name__ == '__main__':
    app = get_app()
    app.run_server(debug=True)

