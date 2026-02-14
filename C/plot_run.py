import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import *
from plotly.subplots import make_subplots
from tqdm import tqdm

INT = np.s_[1:-1, 1:-1]
XINT = np.s_[2:, 1:-1]
XMINT = np.s_[:-2, 1:-1]
YINT = np.s_[1:-1, 2:]
YMINT = np.s_[1:-1, :-2]


def compute_divergence(u):
    divergence = np.zeros_like(u)
    divergence[INT] = (u[XINT] - u[XMINT]) / 2 + (u[YINT] - u[YMINT]) / 2
    return divergence


class RunPlot:
    def __init__(self, re, K, N, PLOTLY_THEME) -> None:
        self.re = re
        self.K = K
        self.N = N
        self.dn = 1 / self.N
        self.PLOTLY_THEME = PLOTLY_THEME

    def prepare(self):
        re_lb = str(int(self.re))
        self.fig = go.FigureWidget(make_subplots(rows=2, cols=2))

        self.fig.add_scatter(y=[], mode="lines", name="min", row=1, col=1)
        self.fig.add_scatter(y=[], mode="lines", name="mean", row=1, col=1)
        self.fig.add_scatter(y=[], mode="lines", name="max", row=1, col=1)

        yL = np.array(list(range(self.N))[::-1]) * self.dn / 1.0
        self.fig.add_scatter(x=yL, y=[], mode="lines", name="profile", row=2, col=1)

        df = pd.read_csv("../ghia_ref_u.csv", sep="\\s+")
        if re_lb in df.columns:
            self.fig.add_scatter(
                x=df["y"],
                y=df[re_lb],
                mode="lines",
                name=f"Ghia & al re={re_lb}",
                row=2,
                col=1,
            )

        self.fig.update_layout(
            template=self.PLOTLY_THEME,
            height=500,
            width=1500,
            font=dict(size=13),
            margin=dict(l=55, r=20, t=15, b=45),
            xaxis=dict(title="Iteration", showgrid=True),
            yaxis=dict(title="Divergence", showgrid=True),
        )

        self._min_div = []
        self._mean_div = []
        self._max_div = []
        self._iter = []

        self.fig.update_traces(line=dict(width=2))
        display(self.fig)

    def update(self, new_frame, step):

        self._iter.append(step)

        div = compute_divergence(new_frame)

        self._min_div.append(div.min())
        self._mean_div.append(div.mean())
        self._max_div.append(div.max())
        half_x = self.N // 2 - 1
        uU = new_frame[half_x, :, 0] / self.K
        with self.fig.batch_update():
            self.fig.data[0].y = self._min_div
            self.fig.data[0].x = self._iter
            self.fig.data[1].y = self._mean_div
            self.fig.data[1].x = self._iter
            self.fig.data[2].y = self._max_div
            self.fig.data[2].x = self._iter

            self.fig.data[3].y = uU
