import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(equity_curve: pd.Series):
    fig, ax = plt.subplots()
    ax.plot(equity_curve.index, equity_curve.values)
    ax.set_title('Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    return fig

def plot_drawdown(equity_curve: pd.Series):
    drawdown = equity_curve / equity_curve.cummax() - 1
    fig, ax = plt.subplots()
    ax.plot(drawdown.index, drawdown.values)
    ax.set_title('Drawdown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    return fig

def plot_signals(price_series: pd.Series, positions: pd.Series):
    fig, ax = plt.subplots()
    ax.plot(price_series.index, price_series.values, label='Price')
    buys  = positions.diff() == 1
    sells = positions.diff() == -1
    ax.scatter(price_series.index[buys],  price_series.values[buys],  marker='^', label='Buy',  s=50)
    ax.scatter(price_series.index[sells], price_series.values[sells], marker='v', label='Sell', s=50)
    ax.set_title('Price with Trade Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

def plot_heatmap(results_df: pd.DataFrame,
                 xlabel: str = 'Exit Threshold',
                 ylabel: str = 'Entry Threshold',
                 title: str = 'Performance Heatmap'):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        results_df.columns.astype(float),
        results_df.index.astype(float),
        results_df.values,
        shading='auto'
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    return fig
