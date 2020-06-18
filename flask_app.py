import flask
from flask import render_template
import dash
import dash_html_components as html
import imbalanceddata
import samplingbias
import overfitting_class
import overfitting_regress
app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

app_0 = dash.Dash(__name__, server=app, url_base_pathname='/app1/')
app_0.layout = html.H1('Under Construction')

app_2 = imbalanceddata.get_app(app)
app_3 = samplingbias.get_app(app)
app_4 = overfitting_class.get_app(app)
app_5 = overfitting_regress.get_app(app)

if __name__ == '__main__':
    app.run()