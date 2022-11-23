from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.io import wavfile as wav
from hmmlearn.hmm import GaussianHMM
from audio.audio_processing import find_calls, get_spectrum

from os import listdir
from os.path import isfile, join

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

def define_slidemarks(spectrum, audio):
    processed_audio = audio[:, 0].astype('f')/1000 # take first channel, scale values down by 1000.
    MAX_T = (len(processed_audio)/spectrum)/60
    SLIDE_MARKS = {i: f'{i/60}m' for i in np.linspace(0, MAX_T*60, 10)}
    return SLIDE_MARKS, MAX_T*60

def get_audio_files(base_dir='../data/'):
    # TODO automatically get from the folder and populate
    onlyfiles = [base_dir + f for f in listdir(base_dir) if (isfile(join(base_dir, f)) and f.endswith('.wav'))]
    return onlyfiles

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children='Splitter Vis'),
        html.H3(children='Audio File'),
        dcc.Dropdown(id='audio-dd'),
        html.Label('Range:'),
        dcc.Slider(
                0, 1,
                value=0, included=False, id='slider')]),

    html.Div(children=[
        dcc.Graph(id='viz-graph')
    ])
])

@app.callback(
    [
        Output("viz-graph", "figure"),
        Output("slider", "marks"), 
        Output("slider", "max"),
        Output("audio-dd", "options")
    ],
    [
        Input("slider", "value"),
        Input("audio-dd", "value")
    ]
)
def update_initial_exposed(start_time, audio_file_name):
    DEFAULT_HMM = GaussianHMM(2, params="st", init_params="st")
    DEFAULT_HMM.means_ = np.array([[0.0], [20.0]])
    DEFAULT_HMM.covars_ = np.array([1e-10, 100]).reshape(2, 1)
    SEGM_LEN=10

    start_time /= 60
    if audio_file_name is None:
        print("Audio file name was none!")
        audio_file_name = '../data/10MinSample.wav'
    spectrum, audio = wav.read(audio_file_name)

    t, f, S = get_spectrum(start_time, spectrum=spectrum, audio=audio, segment_length=SEGM_LEN)
    slidemarks, t_max = define_slidemarks(spectrum, audio)
    options = get_audio_files()

    fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (min)', y='Freq (Hz)'))

    for segment in find_calls(S, f, hmm=DEFAULT_HMM):
        x0, x1 = segment
        x0 = t[x0]; x1 = t[x1]
        if start_time < x0 and start_time + SEGM_LEN/60 > x1:
            fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")

    return fig, slidemarks, t_max, options

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
