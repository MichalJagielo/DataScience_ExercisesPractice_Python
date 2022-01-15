import dash
import dash_core_components as dcc
from dash import html
import plotly.graph_objects as go

app = dash.Dash(__name__)

app.layout = html.Div(children=[

    html.H2(children='Hello App!'),

    dcc.Graph(
        figure=go.Figure([
            go.Bar(
                x=['2017', '2018', '2019'],
                y=[150, 180, 220],
                name='local'
            ),
            go.Bar(
                x=['2017', '2018', '2019'],
                y=[80, 160, 240],
                name='online'
            )
        ])
    )

])

if __name__ == '__main__':
    app.run_server(debug=True)