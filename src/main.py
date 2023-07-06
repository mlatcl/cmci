
import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from call_finder_rnn_simple import CallFinder, load_audio, SR as rnn_sr
from audio.audio_processing import get_spectrum, load_audio_file
import pandas as pd

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import os
from glob import glob
from os import listdir
from os.path import isfile, join

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

CALL_FINDER = CallFinder()

calls = pd.read_excel('../data/calls_for_ml/Calls_ML_Fix.xlsx')
calls.columns = [c.lower().replace(' ', '_') for c in calls.columns]
calls['file'] += '.wav'
calls = calls[['file', 'call_type', 'start', 'end']]
calls = calls.loc[~calls.call_type.isna()]

calls_blackpool = pd.read_excel('../data/calls_for_ml/Blackpool_Labels.xlsx')
calls_blackpool = calls_blackpool.loc[~calls_blackpool.Call_Type.isna(), ['File', 'Call_Type', 'Start', 'End']]
calls_blackpool['File'] = 'Blackpool_Combined_FINAL.wav'
calls_blackpool.columns = calls_blackpool.columns.str.lower()

calls = pd.concat([calls, calls_blackpool], axis=0).reset_index(drop=True)

def define_slidemarks(sampling_rate, audio_len):
    max_time = int(audio_len/sampling_rate)
    slidemarks = {i: f'{np.round(i/60, 1)}m' for i in np.linspace(0, max_time, 10)}
    return slidemarks, max_time

def get_audio_files(base_dir='../data/'):
    result = [y for x in os.walk(base_dir) for y in glob(os.path.join(x[0], '*.wav'))]
    return result

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children='Call Segmentation Visualization'),
        # html.H3(children='Audio File'),
        dcc.Dropdown(id='audio-dd'),
        html.Label('Position along audio file:'),
        dcc.Slider(0, 1, value=0, included=False, id='slider')
    ]),

    html.Div(children=[
        dcc.Graph(id='viz-graph')
    ]),

    html.Div(children=[
        dcc.Graph(id='threshold-graph')
    ]),

    html.Div(children=[
        dcc.Graph(id='features-graph')
    ])
])

@app.callback(
    [
        Output("viz-graph", "figure"),
        Output("threshold-graph", "figure"),
        Output("features-graph", "figure"),
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
    """
    Start time is in seconds.
    """
    print("Start time: {}".format(start_time))
    segment_length = 10 # seconds, width of the spectrum we find_calls over and

    if audio_file_name is None:
        audio_file_name = '../data/calls_for_ml/ML_Test.wav'

    sampling_rate, audio = load_audio_file(audio_file_name)
    S, f, t = get_spectrum(start_time=start_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)

    slidemarks, t_max = define_slidemarks(sampling_rate, len(audio))
    options = get_audio_files(base_dir='../data/calls_for_ml/')

    spectrum_fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (sec)', y='Freq (Hz)'), color_continuous_scale='greys')

    # from IPython.core.debugger import set_trace; set_trace()

    _audio = load_audio(audio_file_name)
    _t = np.arange(len(_audio))/rnn_sr
    # segments, thresholded_spectrum, feature, final_feature = CALL_FINDER.find_calls(S, f, t, smoothing=1300)
    segments, final_feature = CALL_FINDER.find_calls_rnn(_audio[(_t >= start_time) & (_t < start_time + segment_length)], threshold=0.2)

    segments += start_time

    # from IPython.core.debugger import set_trace; set_trace()

    # thresholded_spectrum_fig = px.imshow(thresholded_spectrum, aspect='auto', x=t, y=f, origin='lower',
    #     labels=dict(x='Time (sec)', y='Freq (Hz)'))
    
    # features_fig = px.line(x=t, y=feature)

    half_len_fs = len(f)//2

    for segment in segments:
        x0, x1 = segment
        if start_time < x0 and start_time + segment_length > x1:
            spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[half_len_fs], opacity=0.25, fillcolor="Green")
            # thresholded_spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")

    true_segments = calls.loc[
        (calls.file == audio_file_name.split('/')[-1]) &
        (calls.start >= start_time) & (calls.start <= start_time + segment_length),
        ['start', 'end']]

    for i in range(len(true_segments)):
        x0, x1 = true_segments.iloc[i]
        if start_time < x0 and start_time + segment_length > x1:
            spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[half_len_fs], y1=f[-1], opacity=0.25, fillcolor="Blue")

    return spectrum_fig, None, None, slidemarks, t_max, options

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
