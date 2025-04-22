from .data_loader import download_price_data, load_cached_data
from .polynomial_fit import smooth_price, get_slope, get_acceleration, fit_polynomial, eval_polynomial, get_polynomial_features
from .signal_logic import generate_signals
from .backtester import backtest, generate_trade_log
from .optimizer import optimize_thresholds, get_optimal_thresholds
from .metrics import sharpe_ratio, max_drawdown, win_rate, trade_frequency
from .plots import plot_equity_curve, plot_drawdown, plot_signals, plot_heatmap
