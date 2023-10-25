
import os, sys
import numpy as np
import pandas as pd
from glob import glob
import plotly.express as px
from scipy.io import wavfile as wav

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import torch
from call_finder_rnn_simple import CallFinder as CF_RNN, device, SR as SR_RNN, load_audio as load_torch_audio

from audio.audio_processing import get_spectrum, load_audio_file
from classification_expt import ClassifierPipeline

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

cv_rnn = CF_RNN()

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
                dbc.Row(
                    [
                        html.H2("Audio File"),
                        dcc.Dropdown(get_audio_files(), id="audio-dd"),
                        html.Label('Range:'),
                        dcc.Slider(0, 1, value=0, included=False, id='slider')
                    ],
                    # width=3
                ),
                dbc.Row(
                    [
                        dcc.Graph(id='viz-graph'),

                        dcc.Input(id='filename-input', type='text', placeholder='Enter filename'),
                        html.Button('Save Audio', id='save-button'),
                        html.Div(id="selected-audio-output"),


                    ],
                    # width=9
                )
            ]
        )
    ],
    fluid=True
)

@app.callback(
    Output("slider", "value"),
    Input("audio-dd", "value")
)
def update_initial_exposed(_):
    return 0.0

# @app.callback(
#     [
#         Output("viz-graph", "figure"),
#         Output("slider", "marks"), 
#         Output("slider", "max"),
#     ],
#     [
#         Input("slider", "value"),
#         Input("audio-dd", "value"),
#     ]
# )
# def update_initial_exposed(start_time, audio_file_name):
#     segment_length = 2 # seconds

#     if audio_file_name is None:
#         audio_file_name = '../data/Calls for ML/labelled_data/Blackpool_Combined_FINAL.wav'

#     sampling_rate, audio = load_audio_file(audio_file_name)

#     S, f, t = get_spectrum(start_time=start_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)

#     slidemarks, t_max = define_slidemarks(sampling_rate, len(audio))

#     spectrum_fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower', labels=dict(x='Time (sec)', y='Freq (Hz)'), color_continuous_scale='greys')

#     torch_audio = load_torch_audio(audio_file_name).to(device)
#     torch_t = torch.arange(len(torch_audio)).to(device)/SR_RNN
#     torch_audio = torch_audio[(torch_t >= start_time) & (torch_t < start_time + segment_length)]
#     segments = cv_rnn.find_calls_rnn(torch_audio, start_time=start_time)

#     for segment in segments:
#         x0, x1 = segment
#         if start_time < x0 and start_time + segment_length > x1:
#             spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Orange")
#     spectrum_fig.update_layout(dragmode='drawrect')

#     return spectrum_fig, slidemarks, t_max

# @app.callback(
#     [
#         Output("selected-audio-output", "children"),
#         Output("filename-input", "value"),
#         Output("viz-graph", "figure"),
#     ],
#     [
#         Input("save-button", "n_clicks"),
#         State("viz-graph", "figure"),
#         State("audio-dd", "value"),
#         State("filename-input", "value")
#     ]
# )
# def save_selected_audio(n_clicks, current_fig, audio_file_name, user_filename):
#     if not n_clicks:
#         return dash.no_update, dash.no_update, dash.no_update

#     if audio_file_name is None:
#         audio_file_name = '../data/Calls for ML/labelled_data/Blackpool_Combined_FINAL.wav'

#     try:
#         user_selection = next((entry for entry in current_fig['layout']['shapes'] if entry.get('type') == 'rect' and entry.get('editable')), None)

#         if not user_selection:
#             return "No valid selection made.", "", current_fig

#         start_time_selected = user_selection['x0']
#         end_time_selected = user_selection['x1']

#         sampling_rate, audio = load_audio_file(audio_file_name)

#         start_sample = int(start_time_selected * sampling_rate)
#         end_sample = int(end_time_selected * sampling_rate)

#         selected_audio_segment = audio[start_sample:end_sample]

#         output_filename = f"{user_filename}.wav" if user_filename else f"selected_{start_time_selected}_{end_time_selected}.wav"
#         wav.write("../data/saved_data/" + output_filename, sampling_rate, selected_audio_segment)

#         # Clear the drawn rectangle by filtering out the editable shapes
#         current_fig['layout']['shapes'] = [x for x in current_fig['layout']['shapes'] if not (x.get('type') == 'rect' and x.get('editable'))]

#         return f"Audio segment saved as {output_filename}.", "", current_fig

#     except Exception as e:
#         print("Error:", str(e))
#         raise e
#         return str(e), "", current_fig


@app.callback(
    [
        Output("selected-audio-output", "children"),
        Output("filename-input", "value"),
        Output("viz-graph", "figure"),
        Output("slider", "marks"),
        Output("slider", "max")
    ],
    [
        Input("save-button", "n_clicks"),
        Input("slider", "value"),
        Input("audio-dd", "value")
    ],
    [
        State("viz-graph", "figure"),
        State("filename-input", "value")
    ]
)
def combined_callback(n_clicks, start_time, audio_file_name, current_fig, user_filename):
    # Identify which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if audio_file_name is None:
        audio_file_name = '../data/Calls for ML/labelled_data/Blackpool_Combined_FINAL.wav'

    file_name_for_zoo = audio_file_name.split('/')[-1]

    if file_name_for_zoo[:2] == '08':
        zoo = 'banham'
    elif file_name_for_zoo[:2] == '21':
        zoo = 'shaldon'
    elif file_name_for_zoo[:6] == '202303':
        zoo = 'exmoor'
    elif file_name_for_zoo[:6] in ['202212', '202301']:
        zoo = 'blackpool'
    else:
        zoo = 'unknown'

    file_name_for_zoo = file_name_for_zoo.strip('.wav')

    # Your original logic for the `update_initial_exposed` function
    if triggered_id in ["slider", "audio-dd"]:

        segment_length = 2 # seconds

        sampling_rate, audio = load_audio_file(audio_file_name)

        S, f, t = get_spectrum(start_time=start_time, sampling_rate=sampling_rate, audio=audio, segment_length=segment_length)

        slidemarks, t_max = define_slidemarks(sampling_rate, len(audio))

        spectrum_fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower', labels=dict(x='Time (sec)', y='Freq (Hz)'), color_continuous_scale='greys')

        torch_audio = load_torch_audio(audio_file_name).to(device)
        torch_t = torch.arange(len(torch_audio)).to(device)/SR_RNN
        torch_audio = torch_audio[(torch_t >= start_time) & (torch_t < start_time + segment_length)]
        segments = cv_rnn.find_calls_rnn(torch_audio, start_time=start_time)

        for segment in segments:
            x0, x1 = segment
            if start_time < x0 and start_time + segment_length > x1:
                spectrum_fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Orange")
        spectrum_fig.update_layout(dragmode='drawrect')

        return dash.no_update, dash.no_update, spectrum_fig, slidemarks, t_max
    
    # Your original logic for the `save_selected_audio` function
    elif triggered_id == "save-button":

        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update

        user_selection = next((entry for entry in current_fig['layout']['shapes'] if entry.get('type') == 'rect' and entry.get('editable')), None)

        if not user_selection:
            return "No valid selection made.", "", current_fig

        start_time_selected = user_selection['x0']
        end_time_selected = user_selection['x1']

        sampling_rate, audio = load_audio_file(audio_file_name)

        start_sample = int(start_time_selected * sampling_rate)
        end_sample = int(end_time_selected * sampling_rate)

        selected_audio_segment = audio[start_sample:end_sample]

        # zoo
        # filename
        # start time (2dp)


        output_filename = f"../data/saved_data/{zoo}_{file_name_for_zoo}_{np.round(start_time_selected, 2)}_" + f"{user_filename}.wav" if user_filename else f"selected_{start_time_selected}_{end_time_selected}.wav"
        wav.write(output_filename, sampling_rate, selected_audio_segment)

        # Clear the drawn rectangle by filtering out the editable shapes
        current_fig['layout']['shapes'] = [x for x in current_fig['layout']['shapes'] if not (x.get('type') == 'rect' and x.get('editable'))]

        return f"Audio segment saved as {output_filename}.", "", current_fig, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
