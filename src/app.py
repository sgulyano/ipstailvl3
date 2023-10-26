import dash
from dash import Dash, html

app = Dash(__name__, use_pages=True)
server = app.server
app.title = "หลักสูตร AI Level 3"

app.layout = html.Div([
    dash.page_container
])

model_act = [
    ('กิจกรรม 3', "/hyperparameter-decisiontree", "assets/img/hyperparam.png", 'การปรับไฮเปอร์พารามิเตอร์ ส่งผลอะไรต่อต้นไม้ตัดสินใจ')
]

data_act = [
    ('กิจกรรม 1', "/samplingbias", "assets/img/samplingbias.png", "อคติในการเลือกตัวอย่าง (Sampling Bias) ให้นักเรียนลองเลือกข้อมูลมาเพียงบางส่วน แล้วดูว่าโมเดลที่ได้ต่างจากการใช้ข้อมูลทั้งหมดมากน้อยแค่ไหน"),
    ('กิจกรรม 2', "/imbalanceddata", "assets/img/imbalanceddata.png", ["จำนวนข้อมูลที่ไม่สมดุลกัน (Imbalanced Data) ให้นักเรียนลองใช้เทคนิค",
                                html.Br(),"1) การสุ่มข้อมูลจากกลุ่มหลักให้มีน้อยลง และ",
                                html.Br(),"2) การสร้างข้อมูลของกลุ่มย่อยให้มีมากขึ้น"]),
    ('กิจกรรม 3', "/overfitting-class", "assets/img/classoverfit.png", 'โมเดลที่เฉพาะเจาะจงเกินไป VS โมเดลที่ง่ายเกินไป สำหรับการจำแนก'),
    ('กิจกรรม 4', "/overfitting-regress", "assets/img/regoverfit.png", 'โมเดลที่เฉพาะเจาะจงเกินไป VS โมเดลที่ง่ายเกินไป สำหรับการถดถอย')
]

activities = {'ต้นไม้ตัดสินใจ':model_act,
              'ผลกระทบของข้อมูลต่อโมเดล':data_act}

home_layout = html.Div([
    html.Div([
        html.H1('กิจกรรมหลักสูตร AI Level 3'),
        html.P('จัดทำโดย สสวท.')
        ], className="p-5 bg-secondary text-white text-center"),

    html.Div([
        html.Section([
            html.H2(title, className="mt-4 h3 pb-2 font-weight-normal border-bottom"),
            html.Ul([
                html.Li([
                    html.Div([
                        html.Img(src=img, className="card-img-top"),
                        html.Div([
                            html.H4(name, className="card-tite"),
                            html.P(desc, className="card-text"),
                            html.A("ทำกิจกรรม", className="btn btn-primary", role="button", href=link)
                        ], className="card-body")
                    ], className="card my-3")
                ], className="list-group-item border-0") 
                for name, link, img, desc in arr
            ], className="list-group list-group-horizontal align-items-stretch flex-wrap")
        ]) for title, arr in activities.items()
        ], className="container"),
])

dash.register_page('home', path='/', layout=home_layout, title="หลักสูตร AI Level 3")

if __name__ == '__main__':
    app.run(debug=False)