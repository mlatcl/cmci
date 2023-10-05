
import os, sys
import numpy as np
import pandas as pd
from glob import glob
import plotly.express as px
from scipy.io import wavfile as wav

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

sys.path.append('src')

import torch
from callfinder import CallFinder as CFv0
from call_finder_rnn_simple import load_audio as load_torch_audio, CallFinder as CF_RNN_SUP, device, Files as Files_v1, SR as SR_RNN
from call_finder_pseudolabelling import CallFinder as CF_SMOL_SL

from audio.audio_processing import get_spectrum, load_audio_file
from classification_expt import ClassifierPipeline

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

cf_v0 = CFv0()
cv_rnn_sup = CF_RNN_SUP()
cv_smol_sl = CF_SMOL_SL()
classifier_pipeline = ClassifierPipeline()

def define_slidemarks(sampling_rate, audio_len):
    max_time = int(audio_len/sampling_rate)
    slidemarks = {i: f'{np.round(i/60, 1)}m' for i in np.linspace(0, max_time, 10)}
    return slidemarks, max_time

def get_audio_files(base_dir='../data/Calls for ML/unlabelled_data/'):
    result = [y for x in os.walk(base_dir) for y in glob(os.path.join(x[0], '*.wav'))]
    return result

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Model Selection"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {"label": "OG Heuristic", "value": "orig"},
                                {"label": "RNN Supervised", "value": "rnn_supervised"},
                                {"label": "RNN Semisupervised (Smol)", "value": "rnn_pseudolabelling_small"},
                            ],
                            value="orig",
                        ),
                        html.H2("Audio File"),
                        dcc.Dropdown(get_audio_files(), id="audio-dd"),
                        html.Label('Range:'),
                        dcc.Slider(0, 1, value=0, included=False, id='slider')
                    ],
                    width=3
                ),
                dbc.Col(
                    [
                        dcc.Graph(id='viz-graph')
                    ],
                    width=9
                )
            ]
        )
    ],
    fluid=True
)

@app.callback(
    [
        Output("viz-graph", "figure"),
        Output("slider", "marks"), 
        Output("slider", "max"),
    ],
    [
        Input("slider", "value"),
        Input("audio-dd", "value"),
        Input("model-dropdown", "value")
    ]
)
def update_initial_exposed(start_time, audio_file_name, model_name):
    segment_length = 10 # seconds

    if audio_file_name is None:
        audio_file_name = '../data/Calls for ML/labelled_data/Blackpool_Combined_FINAL.wav'

    sampling_rate, audio = load_audio_file(audio_file_name)

    try:
        S, f, t = get_spectrum(start_time=start_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)
    except:
        from IPython.core.debugger import set_trace; set_trace()
        S, f, t = get_spectrum(start_time=start_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)

    slidemarks, t_max = define_slidemarks(sampling_rate, len(audio))

    spectrum_fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower', labels=dict(x='Time (sec)', y='Freq (Hz)'), color_continuous_scale='greys')

    if model_name == 'orig':
        segments = cf_v0.find_calls(S, f, t)
    elif model_name == 'rnn_supervised':
        torch_audio = load_torch_audio(audio_file_name).to(device)
        torch_t = torch.arange(len(torch_audio)).to(device)/SR_RNN
        torch_audio = torch_audio[(torch_t >= start_time) & (torch_t < start_time + segment_length)]
        segments = cv_rnn_sup.find_calls_rnn(torch_audio, start_time=start_time)
    elif model_name == 'rnn_pseudolabelling_small':
        torch_audio = load_torch_audio(audio_file_name).to(device)
        torch_t = torch.arange(len(torch_audio)).to(device)/SR_RNN
        torch_audio = torch_audio[(torch_t >= start_time) & (torch_t < start_time + segment_length)]
        segments = cv_smol_sl.find_calls_rnn(torch_audio, start_time=start_time)
    else:
        raise ValueError('unknown model')

    if len(segments) > 0:
        classes = classifier_pipeline.predict(audio_file_name, segments[:, 0], segments[:, 1], data_loc='')
    else:
        classes = np.array([])

    for segment, class_label in zip(segments, classes):
        x0, x1 = segment
        if start_time < x0 and start_time + segment_length > x1:
            spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Orange")
            spectrum_fig.add_annotation(
                x=(x0 + x1) / 2,  # x-coordinate for the center of the shape
                y=(f[0] + f[-1]) / 6,  # y-coordinate for the center of the shape
                text=class_label,  # The label text
                showarrow=False,  # We don't want an arrow for the label
            )

    return spectrum_fig, slidemarks, t_max

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
