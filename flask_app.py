import flask
from flask import render_template
import dash
import dash_html_components as html
# import imbalanceddata
# import samplingbias
app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# app = dash.Dash(
#     __name__,
#     server=server,
#     routes_pathname_prefix='/dash/'
# )

# app.layout = html.Div("My Dash app")

app_1 = dash.Dash(__name__, server=app, url_base_pathname='/app1/')
app_1.layout = html.H1('App 1')


app_2 = dash.Dash(__name__, server=app, url_base_pathname='/app2/')
app_2.layout = html.H1('App 2')

# app3 = imbalanceddata.get_app(server)
# app4 = samplingbias.get_app(server)

if __name__ == '__main__':
    app.run(debug=True)

# import dash
# import dash_html_components as html
# from flask import Flask

# server = Flask(__name__)

# # Set-up endpoint 1
# app_1 = dash.Dash(__name__, server=server, url_base_pathname='/app1/')
# app_1.layout = html.H1('App 1')

# # Set-up endpoint 2
# app_2 = dash.Dash(__name__, server=server, url_base_pathname='/app2/')
# app_2.layout = html.H1('App 2')

# # Run server
# server.run()