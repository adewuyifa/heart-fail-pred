import joblib

import dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np


# App dev
navbar = dbc.NavbarSimple(
    brand="Heart Failure Predictor",
    brand_href="https://heart-fail-pred.herokuapp.com",
    brand_style={'fontSize': 30, 'fontWeight': 'bold'}
)

question_one = html.Div(
    [
        dbc.Label("1. What is the Age of the patient?"),
        dbc.Input(
            id='age',
            type='number',
            min=0,
            max=100,
            step=1,
            placeholder="Age in years",
        ),
    ]
)

question_two = html.Div(
    [
        dbc.Label("2. What is the sex of the patient?"),
        dcc.Dropdown(
            id='sex',
            options=[{'label': 'Male', 'value': 'M'},
                     {'label': 'Female', 'value': 'F'}],
            placeholder="Patient's sex",
        ),
    ]
)

question_three = html.Div(
    [
        dbc.Label("3. Chest Pain Type?"),
        dcc.Dropdown(
            id='chest_pain',
            options=[{'label': 'Typical Angina', 'value': 'TA'},
                     {'label': 'Atypical Angina', 'value': 'ATA'},
                     {'label': 'Non-Anginal Pain', 'value': 'NAP'},
                     {'label': 'Asymptomatic', 'value': 'ASY'}],
            placeholder="Patient's chest pain type",
        ),
    ]
)

question_four = html.Div(
    [
        dbc.Label("4. What is the patient's resting blood pressure?"),
        dbc.Input(
            id='bp',
            type='number',
            min=1,
            max=250,
            step=1,
            placeholder="Blood pressure [mm Hg]",
        ),
    ]
)

question_five = html.Div(
    [
        dbc.Label("5. Patient's serum cholesterol?"),
        dbc.Input(
            id='cholesterol',
            type='number',
            min=0,
            max=100,
            step=1,
            placeholder="Cholesterol [mm/dl]",
        ),
    ]
)

question_six = html.Div(
    [
        dbc.Label("6. Patient's Fasting Blood sugar"),
        dcc.Dropdown(
            id='blood_sugar',
            options=[{'label': '> 120 mg/dl', 'value': 1},
                     {'label': 'Otherwise', 'value': 0}],
            placeholder="Fasting BS"
        ),
    ]
)

question_seven = html.Div(
    [
        dbc.Label("7. Patient's resting electrocardiogram results"),
        dcc.Dropdown(
            id='resting_ecg',
            options=[{'label': 'Normal', 'value': 'Normal'},
                     {'label': 'Having ST-T wave abnormality '
                               '(T wave inversions and/or ST elevation or depression of > 0.05 mV)',
                      'value': 'ST'},
                     {'label': "Showing probable or definite left ventricular "
                               "hypertrophy by Estes' criteria", 'value': 'LVH'}],
            optionHeight=50,
            placeholder="RestingECG"
        ),
    ]
)

question_eight = html.Div(
    [
        dbc.Label("8. Maximum heart rate achieved by patient?"),
        dbc.Input(
            id='maximum_hr',
            type='number',
            min=50,
            max=210,
            step=1,
            placeholder="MaxHR",
        ),
    ]
)

question_nine = html.Div(
    [
        dbc.Label("9. Exercise-induced angina"),
        dcc.Dropdown(
            id='exercise_angina',
            options=[{'label': 'Yes', 'value': 'Y'},
                     {'label': 'No', 'value': 'N'}],
            placeholder="ExerciseAngina"
        ),
    ]
)

question_ten = html.Div(
    [
        dbc.Label("10. Oldpeak?"),
        dbc.Input(
            id='oldpeak',
            type='float',
            min=0,
            max=100,
            step=0.1,
            placeholder="Numeric value measured in depression",
        ),
    ]
)

question_eleven = html.Div(
    [
        dbc.Label("11. The slope of the peak exercise ST segment?"),
        dcc.Dropdown(
            id='st_slope',
            options=[{'label': 'Upsloping', 'value': 'Up'},
                     {'label': 'Flat', 'value': 'Flat'},
                     {'label': 'Downsloping', 'value': 'Down'}],
            placeholder="ST slope"
        ),
        html.Br(),
        dbc.Button('Submit',
                   id='submit-val',
                   n_clicks=0,
                   color='primary',
                   className="d-grid gap-2 col-3 mx-auto")
    ]
)

form_1 = dbc.Form([question_one, html.Br(), question_two, html.Br(), question_three])
form_2 = dbc.Form([question_four, html.Br(), question_five, html.Br(), question_six])
form_3 = dbc.Form([question_seven, html.Br(), question_eight, html.Br(), question_nine])
form_4 = dbc.Form([question_ten, html.Br(), question_eleven])

card = dbc.Card(
    [
        dbc.CardBody(
            dbc.Tabs(
                [
                    dbc.Tab(label="Page 1", children=[html.Br(), form_1]),
                    dbc.Tab(label="Page 2", children=[html.Br(), form_2]),
                    dbc.Tab(label="Page 3", children=[html.Br(), form_3]),
                    dbc.Tab(label="Page 4", children=[html.Br(), form_4]),
                ],
                id="card-tabs",
                className="nav nav-pills nav-fill"
            )
        ),
    ]
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
server = app.server
app.config.suppress_callback_exceptions = True
app.title = "Heart Failure Predictor"

app.layout = html.Div([
    html.Div(navbar),
    html.Br(),
    dbc.Container(
        html.H5("Use this machine learning model for early detection of heart failure in patients "
                "with cardiovascular disease or who are at high cardiovascular risk.",
                className="text-center"),
    ),
    html.Hr(),
    dbc.Container([
        dbc.Row([
            dbc.Col(card),
            dbc.Col(id="output_display", style={'height':'10%', 'width':'10%'})
        ])
    ]),
    dcc.Store(id="store-data", data=[], storage_type="session")
])


@app.callback(
    Output("store-data", "data"),
    Input("age", "value"),
    Input("sex", "value"),
    Input("chest_pain", "value"),
    Input("bp", "value"),
    Input("cholesterol", "value"),
    Input("blood_sugar", "value"),
    Input("resting_ecg", "value"),
    Input("maximum_hr", "value"),
    Input("exercise_angina", "value"),
    Input("oldpeak", "value"),
    Input("st_slope", "value")
)
def store_data(age, sex, chest_pain, bp, cholesterol, blood_sugar, resting_ecg,
               maximum_hr, exercise_angina, oldpeak, st_slope):

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": bp,
        "Cholesterol": cholesterol,
        "FastingBS": blood_sugar,
        "RestingECG": resting_ecg,
        "MaxHR": maximum_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }
    df = pd.DataFrame(data, index=[0])

    return df.to_dict('records')


@app.callback(
    Output("output_display", "children"),
    Input("submit-val", "n_clicks"),
    State("store-data", "data")
)
def make_prediction(n_clicks, data):
    if data is None:
        raise PreventUpdate
    if n_clicks > 0:
        df = pd.DataFrame(data, index=[0])
        my_dict = {
            0: "Patient is not at risk of heart failure",
            1: "Patient is at risk of heart failure"
        }
        
        model = joblib.load('finalmodel.pkl')
        prediction = model.predict(df)

        if prediction > 0:
            return dbc.Card([
                dbc.CardImg(src="assets/images/unhealthy-heart.jpg", top=True, style={"width": '45%'}),
                dbc.CardBody(
                    html.H5(f"{np.vectorize(my_dict.get)(prediction[0])}",
                            className="card-text")
                ),
            ], className="g-0 d-flex align-items-center")
        else:
            return dbc.Card([
                dbc.CardImg(src="assets/images/Healthy-heart.jpg", top=True, style={"width": '45%'}),
                dbc.CardBody(
                    html.H5(f"{np.vectorize(my_dict.get)(prediction[0])}",
                            className="card-text")
                ),
            ], className="g-0 d-flex align-items-center")


if __name__ == '__main__':
    app.run_server(debug=False)
