
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import stft
from scipy.io import wavfile as wav
from hmmlearn.hmm import GaussianHMM

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI])

SR, AUDIO = wav.read('monke_data/10MinSample.wav')
AUDIO = AUDIO[:, 0].astype('f')/1000
MAX_T = (len(AUDIO)/SR)/60

SEGM_LEN = 10 # seconds

SLIDE_MARKS = {i: f'{i/60}m' for i in np.linspace(0, MAX_T*60, 10)}

def get_spectrum(start_time):
    start_idx = min(int(SR * start_time * 60), (MAX_T*60 - SEGM_LEN - 1)*SR)
    end_idx = int(SR * (start_time * 60 + SEGM_LEN))
    f, t, spectrum = stft(AUDIO[start_idx:end_idx], nperseg=SR//10, fs=SR)
    return start_time + t/60, f, np.log(np.abs(spectrum) + 1e-10)

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children='Splitter Vis'),

        html.Label('Range:'),
        dcc.Slider(
                0, int(MAX_T*60), 1, marks=SLIDE_MARKS,
                value=0, included=False, id='slider')]),

    html.Div(children=[
        dcc.Graph(id='viz-graph')
    ])
])

hmm = GaussianHMM(2, params="st", init_params="st")
hmm.means_ = np.array([[0.0], [20.0]])
hmm.covars_ = np.array([1e-10, 100]).reshape(2, 1)

def find_calls(S, f):
    s_min, s_max = S.min(), S.max()
    S = (S - s_min) / (s_max - s_min) # normalize
    S = S > 0.8 # maybe change this to quantile
    S[(f < 4e3) | (f > 7000), :] = 0.0

    hmm.fit(S.sum(axis=0).reshape(-1, 1))
    labels = []
    try:
        labels = hmm.predict(S.sum(axis=0).reshape(-1, 1))
        labels = np.diff(labels, prepend=0)

        starts = np.where(labels == 1)[0]
        ends = np.where(labels == -1)[0]

        if (len(starts) != len(ends)):
            if labels[np.where(labels != 0)[0][0]] == 1:
                # call identified at the end and doesn't finish
                ends = np.hstack([ends, len(labels) - 1])
            else:
                starts = np.hstack([0, starts])

        if (len(starts) != len(ends)) or (ends < starts).any():
            from IPython.core.debugger import set_trace; set_trace()
            print('find_calls has failed.')
        labels = np.vstack((starts, ends)).T
    except:
        from IPython.core.debugger import set_trace; set_trace()
    return labels

@app.callback(
    Output("viz-graph", "figure"),
    [Input("slider", "value")]
)
def update_initial_exposed(start_time):
    start_time /= 60
    t, f, S = get_spectrum(start_time)

    fig = px.imshow(S, aspect='auto', x=t, y=f, origin='lower',
        labels=dict(x='Time (min)', y='Freq (Hz)'))

    for segment in find_calls(S, f):
        x0, x1 = segment
        x0 = t[x0]; x1 = t[x1]
        if start_time < x0 and start_time + SEGM_LEN/60 > x1:
            fig.add_shape(x0=x0, x1=x1, y0=f[0], y1=f[-1], opacity=0.25, fillcolor="Green")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
