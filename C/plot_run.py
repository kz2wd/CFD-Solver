import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import *
from plotly.subplots import make_subplots

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
        self.fig = go.FigureWidget(make_subplots(rows=1, cols=2))

        self.fig.add_scatter(y=[], mode="lines", name="min", row=1, col=1)
        self.fig.add_scatter(y=[], mode="lines", name="mean", row=1, col=1)
        self.fig.add_scatter(y=[], mode="lines", name="max", row=1, col=1)

        yL = np.array(list(range(self.N))[::-1]) * self.dn / 1.0
        self.fig.add_scatter(x=yL, y=[], mode="lines", name="profile", row=1, col=2)

        df = pd.read_csv("../ghia_ref_u.csv", sep="\\s+")
        if re_lb in df.columns:
            self.fig.add_scatter(
                x=df["y"],
                y=df[re_lb],
                mode="lines",
                name=f"Ghia & al re={re_lb}",
                row=1,
                col=2,
            )

        self.fig.update_traces(line=dict(width=2))

        self.fig2 = go.FigureWidget(make_subplots(rows=1, cols=2))
        self.fig2.add_trace(
            go.Heatmap(
                z=np.zeros((self.N, self.N)),
                colorscale="Viridis",
                zsmooth=False,
                colorbar=dict(title="Velocity norm", x=0.45),
            ),
            row=1,
            col=1,
        )
        self.fig2.add_trace(
            go.Heatmap(
                z=np.zeros((self.N, self.N)),
                colorscale="Viridis",
                zsmooth=False,
                colorbar=dict(title="Pressure", x=1.00),
            ),
            row=1,
            col=2,
        )
        self.fig2.update_xaxes(scaleanchor="y")
        self.fig2.update_yaxes(constrain="domain")
        self.fig2.update_yaxes(autorange="reversed")
        self.fig.update_layout(
            template=self.PLOTLY_THEME,
            height=300,
            width=1400,
            font=dict(size=13),
            margin=dict(l=55, r=20, t=15, b=45),
            xaxis=dict(title="Iteration", showgrid=True),
            yaxis=dict(title="Divergence", showgrid=True),
        )
        self.fig2.update_layout(template=self.PLOTLY_THEME, height=400, width=1400)

        self._min_div = []
        self._mean_div = []
        self._max_div = []
        self._iter = []

        display(self.fig)
        display(self.fig2)

    def update(self, velocity, pressure, step):

        self._iter.append(step)

        div = compute_divergence(velocity)

        self._min_div.append(div.min())
        self._mean_div.append(div.mean())
        self._max_div.append(div.max())
        half_x = self.N // 2 - 1
        uU = velocity[half_x, :, 0] / self.K
        with self.fig.batch_update():
            self.fig.data[0].y = self._min_div
            self.fig.data[0].x = self._iter
            self.fig.data[1].y = self._mean_div
            self.fig.data[1].x = self._iter
            self.fig.data[2].y = self._max_div
            self.fig.data[2].x = self._iter

            self.fig.data[3].y = uU

        with self.fig2.batch_update():
            z0 = np.linalg.norm(np.swapaxes(velocity, 0, 1), axis=-1)
            z1 = np.swapaxes(np.squeeze(pressure, axis=-1), 0, 1)

            self.fig2.data[0].update(z=z0)
            self.fig2.data[0].zauto = True
            self.fig2.data[1].update(z=z1)
            self.fig2.data[1].zauto = True
            # self.fig2.data[0].z = z0
            # self.fig2.data[0].zmin = z0.min()
            # self.fig2.data[0].zmax = z0.max()
        
            # self.fig2.data[1].z = z1
            # self.fig2.data[1].zmin = z1.min()
            # self.fig2.data[1].zmax = z1.max()
