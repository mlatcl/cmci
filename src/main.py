
import numpy as np
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from callfinder import CallFinder
from audio.audio_processing import get_spectrum

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from os import listdir
from os.path import isfile, join

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

CALL_FINDER = CallFinder()

def define_slidemarks(sampling_rate, audio_len):
    max_time = int(audio_len/sampling_rate)
    slidemarks = {i: f'{np.round(i/60, 1)}m' for i in np.linspace(0, max_time, 10)}
    return slidemarks, max_time

def get_audio_files(base_dir='../data/'):
    onlyfiles = [base_dir + f \
        for f in listdir(base_dir) \
        if (isfile(join(base_dir, f)) and f.endswith('.wav')) \
    ]
    return onlyfiles

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children='Splitter Vis'),
        html.H3(children='Audio File'),
        dcc.Dropdown(id='audio-dd'),
        html.Label('Range:'),
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
    start_time /= 60
    print("Start time: {}".format(start_time))
    segment_length = 10 # seconds, width of the spectrum we find_calls over and

    if audio_file_name is None:
        audio_file_name = '../data/10MinSample.wav'

    sampling_rate, audio = wav.read(audio_file_name)
    t, f, S = get_spectrum(start_time=start_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)

    slidemarks, t_max = define_slidemarks(sampling_rate, len(audio))
    options = get_audio_files()

    spectrum_fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (min)', y='Freq (Hz)'))

    segments, thresholded_spectrum, features, final_feature = CALL_FINDER.find_calls(S, f, t)

    thresholded_spectrum_fig = px.imshow(thresholded_spectrum, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (min)', y='Freq (Hz)'))
    
    features_fig = px.line(x=t, y=features[:,0], range_y=(-1000, 1000))
    # features_fig = px.line(x=t, y=final_feature[:,0], range_y=(-100, 100))
    for i in range(0,features.shape[1]):
        features_fig.add_scatter(x=t, y=features[:,i], mode='lines', name=f'Feature {i}')

    print("Final Feature Outside: {}".format(final_feature))
    features_fig.add_scatter(x=t, y=final_feature[:,0], mode='lines', name='Final Feature')
    


    for segment in segments:
        x0, x1 = segment
        if start_time < x0 and start_time + segment_length/60 > x1:
            spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")
            thresholded_spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")

    return spectrum_fig, thresholded_spectrum_fig, features_fig, slidemarks, t_max, options

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

## Notes
"""
Notes:

Have a way to change parameters like threshold quantile and measure how many calls we find / (how many we (miss?))
"""