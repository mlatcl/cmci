
import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from utils import CallFinder

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

CALL_FINDER = CallFinder()

SR, AUDIO = wav.read('monke_data/10MinSample.wav')
AUDIO = AUDIO[:, 0].astype('f')/1000

freqs, timepoints, spectrum = stft(AUDIO, nperseg=SR//10, fs=SR)
timepoints /= 60
spectrum = np.log(np.abs(spectrum) + 1e-10)

SEGM_LEN = 10/60 # minutes
MAX_T = (len(AUDIO)/SR)/60
SLIDE_MARKS = {i: f'{i/60}m' for i in np.linspace(0, MAX_T*60, 10)}

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children='Splitter Vis'),

        html.Label('Range:'),
        dcc.Slider(0, int(MAX_T*60), 1, marks=SLIDE_MARKS,
                value=0, included=False, id='slider')]),

    html.Div(children=[
        dcc.Graph(id='viz-graph')
    ])
])

@app.callback(
    Output("viz-graph", "figure"),
    [Input("slider", "value")]
)
def update_initial_exposed(start_time):
    start_time /= 60
    idxs = np.where((timepoints >= start_time) & (timepoints <= start_time + SEGM_LEN))[0]
    t, f, S = timepoints[idxs], freqs, spectrum[:, idxs]

    fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (min)', y='Freq (Hz)'))

    for segment in CALL_FINDER.find_calls(S, f, t):
        x0, x1 = segment
        if start_time < x0 and start_time + SEGM_LEN > x1:
            fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
