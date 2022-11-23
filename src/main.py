
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from hmmlearn.hmm import GaussianHMM
from audio.audio_processing import find_calls

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

SR, AUDIO = wav.read('../data/10MinSample.wav')
AUDIO = AUDIO[:, 0].astype('f')/1000
MAX_T = (len(AUDIO)/SR)/60

SEGM_LEN = 10 # seconds

SLIDE_MARKS = {i: f'{i/60}m' for i in np.linspace(0, MAX_T*60, 10)}

def get_spectrum(start_time):
    start_idx = min(int(SR * start_time * 60), (MAX_T*60 - SEGM_LEN - 1)*SR)
    end_idx = int(SR * (start_time * 60 + SEGM_LEN))
    f, t, spectrum = stft(AUDIO[start_idx:end_idx], nperseg=SR//10, fs=SR)
    return start_time + t/60, f, np.log(np.abs(spectrum) + 1e-10)

AUDIO_FILES=['file1.wav', 'file2.wav', 'file3.wav']

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children='Splitter Vis'),
        html.H3(children='Audio File'),
        dcc.Dropdown(AUDIO_FILES, AUDIO_FILES[0], id='audio-dd'),
        html.Label('Range:'),
        dcc.Slider(
                0, int(MAX_T*60), 1, marks=SLIDE_MARKS,
                value=0, included=False, id='slider')]),

    html.Div(children=[
        dcc.Graph(id='viz-graph')
    ])
])

# @app.callback(
#     # Output(),
#     [Input("audio-dd", "value")]
# )


@app.callback(
    Output("viz-graph", "figure"),
    [Input("slider", "value")]
)
def update_initial_exposed(start_time):
    DEFAULT_HMM = GaussianHMM(2, params="st", init_params="st")
    DEFAULT_HMM.means_ = np.array([[0.0], [20.0]])
    DEFAULT_HMM.covars_ = np.array([1e-10, 100]).reshape(2, 1)

    start_time /= 60
    t, f, S = get_spectrum(start_time)

    fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (min)', y='Freq (Hz)'))

    for segment in find_calls(S, f, hmm=DEFAULT_HMM):
        x0, x1 = segment
        x0 = t[x0]; x1 = t[x1]
        if start_time < x0 and start_time + SEGM_LEN/60 > x1:
            fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
