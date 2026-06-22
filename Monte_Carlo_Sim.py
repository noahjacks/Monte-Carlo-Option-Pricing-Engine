"""
Monte Carlo Option Pricing Engine
==================================
Pure Python — numpy · scipy · matplotlib (no browser, no JS, no server).
 
Run:  python3 monte_carlo_options.py
 
Controls (left panel):
  - Radio buttons  : Call / Put, European / American
  - Text boxes     : all model parameters
  - "Run" button   : price the option and refresh charts
 
Output (right panels):
  - Metric strip   : MC price, std error, confidence interval, BS benchmark
  - Greeks strip   : Δ Γ Θ ν ρ (analytical Black-Scholes)
  - Histogram      : distribution of discounted payoffs across all paths
  - Path chart     : 20 sample GBM trajectories + strike overlay
"""
 
import threading
import numpy as np
from scipy.stats import norm
from scipy.linalg import lstsq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button, RadioButtons
 
# Try interactive backends in order of preference.
# TkAgg works everywhere tkinter is installed (most Python distributions).
# Qt5Agg / Qt6Agg need PyQt5/PySide6.  MacOSX works on macOS without extras.
# If none work the script falls back to whatever matplotlib defaults to.
for _backend in ("TkAgg", "Qt5Agg", "Qt6Agg", "MacOSX", "WXAgg"):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_test   # noqa — just testing the import
        break
    except Exception:
        continue
plt.rcParams.update({
    "figure.facecolor":  "#1a1a1f",
    "axes.facecolor":    "#1a1a1f",
    "text.color":        "#e8e6f0",
    "axes.labelcolor":   "#e8e6f0",
    "xtick.color":       "#888899",
    "ytick.color":       "#888899",
    "axes.edgecolor":    "#333344",
    "grid.color":        "#2a2a35",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
})
 
ACCENT    = "#7c6fff"   # indigo-purple
ACCENT2   = "#a78bfa"
RED_DASH  = "#f87171"
BG        = "#1a1a1f"
BG2       = "#22222a"
PANEL_BG  = "#13131a"
 
 
# ────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ────────────────────────────────────────────────────────────────────────────
 
def black_scholes(S, K, r, q, sigma, T, option_type):
    """Closed-form BS price for a European option."""
    if T <= 0:
        # At expiry the option is worth exactly its intrinsic value
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)   # risk-neutral prob of finishing ITM = N(d2)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
 
 
def compute_greeks(S, K, r, q, sigma, T, option_type):
    """
    Analytical BS Greeks in market-standard units:
      Theta  → per calendar day (÷365)
      Vega   → per 1-vol-point  (×0.01)
      Rho    → per 1-rate-point (×0.01)
    """
    if T <= 0:
        return dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    sqT  = np.sqrt(T)
    d1   = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2   = d1 - sigma * sqT
    nd1  = norm.pdf(d1)
    eqT, erT = np.exp(-q * T), np.exp(-r * T)
 
    # Delta: ∂V/∂S — also roughly the risk-neutral exercise probability
    delta = eqT * norm.cdf(d1)  if option_type == "call" else -eqT * norm.cdf(-d1)
 
    # Gamma: ∂²V/∂S² — identical for calls and puts (put-call parity)
    gamma = (eqT * nd1) / (S * sigma * sqT)
 
    # Theta: time decay. Three terms: vol drag, discounted strike cash flow, dividend effect.
    theta_call = (-(S * eqT * nd1 * sigma) / (2 * sqT)
                  - r * K * erT * norm.cdf(d2)
                  + q * S * eqT * norm.cdf(d1))
    theta_put  = (-(S * eqT * nd1 * sigma) / (2 * sqT)
                  + r * K * erT * norm.cdf(-d2)
                  - q * S * eqT * norm.cdf(-d1))
    theta = (theta_call if option_type == "call" else theta_put) / 365
 
    # Vega: same for calls and puts by put-call parity; scaled to a 1-vol-point move
    vega = S * eqT * nd1 * sqT * 0.01
 
    # Rho: calls benefit from rising rates (higher forward); puts are hurt
    rho = (K * T * erT * norm.cdf(d2) * 0.01  if option_type == "call"
           else -K * T * erT * norm.cdf(-d2) * 0.01)
 
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
 
 
# ────────────────────────────────────────────────────────────────────────────
# GBM path simulation (antithetic variates)
# ────────────────────────────────────────────────────────────────────────────
 
def simulate_gbm(S, r, q, sigma, T, n_sims, n_steps):
    """
    Exact log-return GBM discretization with antithetic variates.
 
    Under the risk-neutral measure:  dS = (r−q)S dt + σS dW
    Log-return per step: (r−q−σ²/2)dt + σ√dt·Z  (Itô correction keeps E[S_T] = S·e^{(r−q)T})
 
    Antithetic variates: pairing z with −z halves variance for ~free.
    Returns price matrix of shape (n_sims, n_steps+1).
    """
    dt    = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)
    half  = n_sims // 2
 
    rng    = np.random.default_rng()
    z      = rng.standard_normal((half, n_steps))
    z_full = np.vstack([z, -z])                      # antithetic pairs
 
    log_ret = drift + vol * z_full
    cum_ret = np.cumsum(log_ret, axis=1)
    return S * np.exp(np.hstack([np.zeros((n_sims, 1)), cum_ret]))
 
 
# ────────────────────────────────────────────────────────────────────────────
# Pricers
# ────────────────────────────────────────────────────────────────────────────
 
def price_european(S, K, r, q, sigma, T, n_sims, n_steps, conf, option_type):
    """Average discounted terminal payoff across all paths."""
    paths    = simulate_gbm(S, r, q, sigma, T, n_sims, n_steps)
    S_T      = paths[:, -1]
    discount = np.exp(-r * T)
    payoffs  = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
 
    price  = discount * payoffs.mean()
    se     = discount * payoffs.std(ddof=1) / np.sqrt(n_sims)
    z_star = norm.ppf(0.5 + conf / 2)          # e.g. 1.96 for 95%
 
    return dict(
        price    = float(price),
        se       = float(se),
        ci_low   = float(price - z_star * se),
        ci_high  = float(price + z_star * se),
        bs_price = float(black_scholes(S, K, r, q, sigma, T, option_type)),
        greeks   = compute_greeks(S, K, r, q, sigma, T, option_type),
        payoffs  = (payoffs * discount).tolist(),
        paths    = paths[:20].tolist(),
    )
 
 
def price_american(S, K, r, q, sigma, T, n_sims, n_steps, conf, option_type):
    """
    Longstaff-Schwartz (LSM) least-squares Monte Carlo for American options.
 
    Backward induction: at each time step, regress discounted future cashflows
    onto {1, S, S²} for in-the-money paths, then exercise early if intrinsic
    value beats estimated continuation value.
    """
    paths     = simulate_gbm(S, r, q, sigma, T, n_sims, n_steps)
    dt        = T / n_steps
    step_disc = np.exp(-r * dt)
 
    # Start from terminal payoff
    S_T       = paths[:, -1]
    cashflows = (np.maximum(S_T - K, 0) if option_type == "call"
                 else np.maximum(K - S_T, 0)).astype(float)
 
    # Roll backward from T−1 to t=1
    for t in range(n_steps - 1, 0, -1):
        S_t       = paths[:, t]
        intrinsic = (np.maximum(S_t - K, 0) if option_type == "call"
                     else np.maximum(K - S_t, 0))
        itm = intrinsic > 0
 
        if itm.sum() > 3:
            # Regress discounted future cash on polynomial basis of current stock price.
            # Only ITM paths matter — OTM paths have no exercise decision to make.
            X          = S_t[itm]
            Y          = cashflows[itm] * step_disc
            basis      = np.column_stack([np.ones_like(X), X, X**2])
            coeffs, *_ = lstsq(basis, Y)
            cont       = basis @ coeffs             # estimated continuation value
 
            # Exercise where intrinsic beats continuation
            exercise          = intrinsic[itm] > cont
            cashflows[itm]    = np.where(exercise, intrinsic[itm], cashflows[itm] * step_disc)
            cashflows[~itm]  *= step_disc
        else:
            cashflows *= step_disc                  # too few ITM paths — just discount
 
    cashflows *= step_disc                          # final discount: t=1 → t=0
    price  = float(cashflows.mean())
    se     = float(cashflows.std(ddof=1) / np.sqrt(n_sims))
    z_star = norm.ppf(0.5 + conf / 2)
 
    return dict(
        price    = price,
        se       = se,
        ci_low   = price - z_star * se,
        ci_high  = price + z_star * se,
        bs_price = None,                            # no simple closed-form for Americans
        greeks   = compute_greeks(S, K, r, q, sigma, T, option_type),
        payoffs  = cashflows.tolist(),
        paths    = paths[:20].tolist(),
    )
 
 
# ────────────────────────────────────────────────────────────────────────────
# GUI
# ────────────────────────────────────────────────────────────────────────────
 
class OptionPricingApp:
    """
    Full application: matplotlib figure with widgets in the left column,
    results and charts in the right column.  All state lives on this object.
    """
 
    # Default parameter values shown on startup
    DEFAULTS = dict(S=100, K=105, r=5, sigma=20, T=1.0,
                    q=0, N=20000, steps=252, conf=95)
 
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 8.5), facecolor=PANEL_BG)
        self.fig.canvas.manager.set_window_title("Monte Carlo Option Pricing Engine")
        self._build_layout()
        self._build_controls()
        self._build_output_area()
        self._run()                                 # auto-price on startup
 
    # ── Layout ───────────────────────────────────────────────────────────────
 
    def _build_layout(self):
        # Outer grid: left control column (28%) | right output column (72%)
        outer = gridspec.GridSpec(1, 2, figure=self.fig,
                                  left=0.01, right=0.99,
                                  top=0.97, bottom=0.03,
                                  wspace=0.04,
                                  width_ratios=[0.28, 0.72])
 
        self.ax_ctrl  = self.fig.add_subplot(outer[0])   # placeholder — turned off
        self.ax_ctrl.set_visible(False)
 
        # Right column: metrics, greeks, histogram, paths
        right = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[1],
            hspace=0.45,
            height_ratios=[0.12, 0.12, 0.38, 0.38],
        )
        self.ax_metrics = self.fig.add_subplot(right[0])
        self.ax_greeks  = self.fig.add_subplot(right[1])
        self.ax_hist    = self.fig.add_subplot(right[2])
        self.ax_paths   = self.fig.add_subplot(right[3])
 
        for ax in (self.ax_metrics, self.ax_greeks):
            ax.set_facecolor(PANEL_BG)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
 
        for ax in (self.ax_hist, self.ax_paths):
            ax.set_facecolor(BG2)
            ax.grid(True, axis="y")
            for spine in ax.spines.values():
                spine.set_color("#333344")
 
    # ── Control panel ────────────────────────────────────────────────────────
 
    def _add_textbox(self, label, default, rect):
        """Helper: labelled text box.  Returns the TextBox widget."""
        ax  = self.fig.add_axes(rect, facecolor=BG2)
        box = TextBox(ax, "", initial=str(default),
                      color=BG2, hovercolor="#2a2a35",
                      label_pad=0.02)
        box.label.set_color("#888899")
        box.text_disp.set_color("#e8e6f0")
        box.text_disp.set_fontsize(9)
        # Place the label above the box
        self.fig.text(rect[0], rect[1] + rect[3] + 0.005, label,
                      fontsize=7.5, color="#888899", va="bottom",
                      transform=self.fig.transFigure)
        return box
 
    def _build_controls(self):
        """Create all widgets in the left column (axes coordinates)."""
        lx  = 0.015   # left edge of control column
        rw  = 0.24    # widget width
        bh  = 0.030   # box height
        col2 = lx + rw / 2 + 0.01   # right sub-column x
 
        # ── Title ──────────────────────────────────────────────────────────
        self.fig.text(lx, 0.95, "Option Pricing", fontsize=11,
                      color=ACCENT2, fontweight="bold",
                      transform=self.fig.transFigure)
        self.fig.text(lx, 0.925, "Monte Carlo Engine", fontsize=8,
                      color="#666677", transform=self.fig.transFigure)
 
        # ── Option type radio ───────────────────────────────────────────────
        self.fig.text(lx, 0.90, "OPTION TYPE", fontsize=7,
                      color="#555566", transform=self.fig.transFigure,
                      fontweight="bold")
        ax_type = self.fig.add_axes([lx, 0.855, rw, 0.042], facecolor=PANEL_BG)
        self.radio_type = RadioButtons(ax_type, ("Call", "Put"),
                                       activecolor=ACCENT)
        for lbl in self.radio_type.labels:
            lbl.set_color("#e8e6f0"); lbl.set_fontsize(9)
 
        # ── Exercise style radio ────────────────────────────────────────────
        self.fig.text(lx, 0.835, "EXERCISE STYLE", fontsize=7,
                      color="#555566", transform=self.fig.transFigure,
                      fontweight="bold")
        ax_style = self.fig.add_axes([lx, 0.790, rw, 0.042], facecolor=PANEL_BG)
        self.radio_style = RadioButtons(ax_style, ("European", "American"),
                                        activecolor=ACCENT)
        for lbl in self.radio_style.labels:
            lbl.set_color("#e8e6f0"); lbl.set_fontsize(9)
 
        # ── Parameter text boxes (two-column layout) ────────────────────────
        self.fig.text(lx, 0.768, "PARAMETERS", fontsize=7,
                      color="#555566", transform=self.fig.transFigure,
                      fontweight="bold")
 
        hw = (rw - 0.01) / 2   # half-width for two-column layout
 
        def row(y):
            """Return rects for left and right cells in a two-column row."""
            return ([lx, y, hw, bh], [lx + hw + 0.01, y, hw, bh])
 
        r0l, r0r = row(0.720)
        r1l, r1r = row(0.672)
        r2l, r2r = row(0.624)
        r3l, r3r = row(0.576)
        r4l, r4r = row(0.528)
 
        self.tb_S     = self._add_textbox("Spot (S)",          self.DEFAULTS["S"],     r0l)
        self.tb_K     = self._add_textbox("Strike (K)",        self.DEFAULTS["K"],     r0r)
        self.tb_r     = self._add_textbox("Rate % (r)",        self.DEFAULTS["r"],     r1l)
        self.tb_sigma = self._add_textbox("Vol % (σ)",         self.DEFAULTS["sigma"], r1r)
        self.tb_T     = self._add_textbox("Expiry yrs (T)",    self.DEFAULTS["T"],     r2l)
        self.tb_q     = self._add_textbox("Div yield % (q)",   self.DEFAULTS["q"],     r2r)
        self.tb_N     = self._add_textbox("Simulations",       self.DEFAULTS["N"],     r3l)
        self.tb_steps = self._add_textbox("Time steps",        self.DEFAULTS["steps"], r3r)
        self.tb_conf  = self._add_textbox("Conf. level %",     self.DEFAULTS["conf"],  r4l)
 
        # ── Run button ──────────────────────────────────────────────────────
        ax_btn = self.fig.add_axes([lx, 0.470, rw, 0.040], facecolor=ACCENT)
        self.btn_run = Button(ax_btn, "Run Simulation",
                              color=ACCENT, hovercolor=ACCENT2)
        self.btn_run.label.set_color("#ffffff")
        self.btn_run.label.set_fontsize(10)
        self.btn_run.label.set_fontweight("bold")
        self.btn_run.on_clicked(self._on_run)
 
        # ── Status text ─────────────────────────────────────────────────────
        self.txt_status = self.fig.text(
            lx, 0.452, "", fontsize=8,
            color="#666677", transform=self.fig.transFigure)
 
        # ── Method note ─────────────────────────────────────────────────────
        note = ("GBM · Euler-Maruyama · antithetic variates\n"
                "American: Longstaff-Schwartz LSM regression\n"
                "Greeks: analytical Black-Scholes")
        self.fig.text(lx, 0.05, note, fontsize=7, color="#444455",
                      transform=self.fig.transFigure, linespacing=1.8)
 
    # ── Output area (metric strip, greeks strip, charts) ─────────────────────
 
    def _metric_text(self, ax, x, label, value, color="#e8e6f0"):
        """Place a label/value pair at x position inside a blank axes."""
        ax.text(x, 0.72, label, transform=ax.transAxes,
                fontsize=7.5, color="#888899", va="top", ha="center")
        ax.text(x, 0.28, value, transform=ax.transAxes,
                fontsize=13, color=color, va="bottom", ha="center",
                fontweight="bold", fontfamily="monospace")
 
    def _build_output_area(self):
        """Draw placeholder metric/greek strips; charts are drawn on first run."""
        self._draw_metrics("—", "—", "—  –  —", "—")
        self._draw_greeks(dict(delta=0, gamma=0, theta=0, vega=0, rho=0), placeholder=True)
 
        # Axis labels
        self.ax_hist.set_title("Payoff distribution",
                                fontsize=9, color="#888899", pad=4)
        self.ax_hist.set_xlabel("Discounted payoff ($)", fontsize=8)
        self.ax_hist.set_ylabel("Frequency",             fontsize=8)
 
        self.ax_paths.set_title("Sample price paths (20 of N)",
                                 fontsize=9, color="#888899", pad=4)
        self.ax_paths.set_xlabel("Time",   fontsize=8)
        self.ax_paths.set_ylabel("Price ($)", fontsize=8)
 
    def _draw_metrics(self, price, se, ci, bs):
        ax = self.ax_metrics
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
 
        # Divider line at top of metrics strip
        ax.axhline(1, color="#333344", linewidth=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
 
        # Four metric blocks evenly spaced
        items = [
            (0.125, "MC PRICE",       price, ACCENT2),
            (0.375, "STD ERROR",      se,    "#e8e6f0"),
            (0.625, "CONF. INTERVAL", ci,    "#e8e6f0"),
            (0.875, "BS PRICE",       bs,    "#e8e6f0"),
        ]
        for x, lbl, val, col in items:
            self._metric_text(ax, x, lbl, val, color=col)
 
        # Subtle vertical dividers between blocks
        for xd in (0.25, 0.50, 0.75):
            ax.axvline(xd, color="#252530", linewidth=0.8)
 
    def _draw_greeks(self, g, placeholder=False):
        ax = self.ax_greeks
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
 
        fmt = "—" if placeholder else None
        greek_items = [
            (0.1,  "Δ Delta",  g["delta"]),
            (0.3,  "Γ Gamma",  g["gamma"]),
            (0.5,  "Θ Theta",  g["theta"]),
            (0.7,  "ν Vega",   g["vega"]),
            (0.9,  "ρ Rho",    g["rho"]),
        ]
        for x, name, val in greek_items:
            label  = name
            value  = fmt if fmt else f"{val:+.4f}"
            color  = "#e8e6f0" if fmt else ("#7cffa0" if val > 0 else RED_DASH if val < 0 else "#888899")
            ax.text(x, 0.72, label, transform=ax.transAxes,
                    fontsize=7.5, color="#888899", va="top", ha="center")
            ax.text(x, 0.28, value, transform=ax.transAxes,
                    fontsize=11, color=color, va="bottom", ha="center",
                    fontweight="bold", fontfamily="monospace")
        for xd in (0.2, 0.4, 0.6, 0.8):
            ax.axvline(xd, color="#252530", linewidth=0.8)
 
    def _draw_histogram(self, payoffs):
        ax = self.ax_hist
        ax.clear()
        ax.set_facecolor(BG2)
        ax.grid(True, axis="y", color="#2a2a35", linewidth=0.5)
        for sp in ax.spines.values(): sp.set_color("#333344")
 
        payoffs_arr = np.array(payoffs)
 
        # Split payoffs into zero (expired OTM) and positive bins
        zero_count = np.sum(payoffs_arr == 0)
        pos        = payoffs_arr[payoffs_arr > 0]
 
        if len(pos) > 0:
            n_bins = 40
            counts, edges = np.histogram(pos, bins=n_bins)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            width = edges[1] - edges[0]
            ax.bar(bin_centers, counts, width=width * 0.95,
                   color=ACCENT, alpha=0.85, linewidth=0)
 
        # Zero-payoff bar on the left with a different color to show OTM mass
        if zero_count > 0:
            ax.bar([-0.5], [zero_count], width=0.4,
                   color="#444460", alpha=0.9, linewidth=0, label="OTM (zero payoff)")
            ax.legend(fontsize=7, loc="upper right",
                      facecolor=PANEL_BG, edgecolor="#333344",
                      labelcolor="#888899")
 
        ax.set_title("Payoff distribution", fontsize=9, color="#888899", pad=4)
        ax.set_xlabel("Discounted payoff ($)", fontsize=8)
        ax.set_ylabel("Frequency",             fontsize=8)
        ax.tick_params(labelsize=8)
 
    def _draw_paths(self, paths, K, T):
        ax = self.ax_paths
        ax.clear()
        ax.set_facecolor(BG2)
        ax.grid(True, axis="y", color="#2a2a35", linewidth=0.5)
        for sp in ax.spines.values(): sp.set_color("#333344")
 
        n_steps = len(paths[0]) - 1
        t_axis  = np.linspace(0, T, n_steps + 1)
 
        for i, path in enumerate(paths):
            hue = 220 + i * 7          # rotate through blue-purple range
            sat = 55
            val = 55
            color = f"hsl({hue},{sat}%,{val}%)"
            # matplotlib doesn't support hsl() strings directly — convert to hex
            import colorsys
            h, s, v = hue / 360, sat / 100, val / 100
            r, g, b = colorsys.hls_to_rgb(h, v, s)
            ax.plot(t_axis, path, color=(r, g, b), linewidth=0.7, alpha=0.7)
 
        # Strike overlay — the horizontal red dashed line
        ax.axhline(K, color=RED_DASH, linewidth=1.2,
                   linestyle="--", label=f"Strike K={K:.0f}")
        ax.legend(fontsize=7, loc="upper left",
                  facecolor=PANEL_BG, edgecolor="#333344",
                  labelcolor="#888899")
 
        ax.set_title("Sample price paths (20 of N)", fontsize=9, color="#888899", pad=4)
        ax.set_xlabel("Time (yrs)", fontsize=8)
        ax.set_ylabel("Price ($)",  fontsize=8)
        ax.tick_params(labelsize=8)
 
    # ── Simulation dispatch ───────────────────────────────────────────────────
 
    def _read_params(self):
        """Parse all text-box inputs; raise ValueError with a clear message if anything is bad."""
        def fv(box, name):
            try:
                return float(box.text)
            except ValueError:
                raise ValueError(f"Invalid value for {name}: '{box.text}'")
 
        S     = fv(self.tb_S,     "Spot")
        K     = fv(self.tb_K,     "Strike")
        r     = fv(self.tb_r,     "Rate")      / 100
        sigma = fv(self.tb_sigma, "Volatility") / 100
        T     = fv(self.tb_T,     "Expiry")
        q     = fv(self.tb_q,     "Dividend")  / 100
        N     = int(fv(self.tb_N,     "Simulations"))
        steps = int(fv(self.tb_steps, "Steps"))
        conf  = fv(self.tb_conf,  "Confidence") / 100
 
        opt_type = self.radio_type.value_selected.lower()    # "call" or "put"
        style    = self.radio_style.value_selected.lower()   # "european" or "american"
 
        return S, K, r, sigma, T, q, N, steps, conf, opt_type, style
 
    def _on_run(self, event=None):
        """Button callback — run simulation in a background thread so the UI stays responsive."""
        self.btn_run.label.set_text("Running…")
        self.txt_status.set_text("Simulating…")
        self.fig.canvas.draw_idle()
 
        # Run the heavy computation in a thread so matplotlib's event loop keeps ticking
        threading.Thread(target=self._run, daemon=True).start()
 
    def _run(self):
        try:
            S, K, r, sigma, T, q, N, steps, conf, opt_type, style = self._read_params()
        except ValueError as e:
            self.txt_status.set_text(str(e))
            self.btn_run.label.set_text("Run Simulation")
            self.fig.canvas.draw_idle()
            return
 
        self.txt_status.set_text(f"Running {N:,} paths × {steps} steps…")
        self.fig.canvas.draw_idle()
 
        try:
            pricer = price_european if style == "european" else price_american
            result = pricer(S, K, r, q, sigma, T, N, steps, conf, opt_type)
        except Exception as e:
            self.txt_status.set_text(f"Error: {e}")
            self.btn_run.label.set_text("Run Simulation")
            self.fig.canvas.draw_idle()
            return
 
        # Format result strings
        price_str = f"${result['price']:.4f}"
        se_str    = f"${result['se']:.4f}"
        ci_str    = f"${result['ci_low']:.2f} – ${result['ci_high']:.2f}"
        bs_str    = f"${result['bs_price']:.4f}" if result["bs_price"] is not None else "N/A"
        conf_pct  = round(conf * 100)
        ci_label  = f"CONF. INTERVAL ({conf_pct}%)"
 
        # Update metric strip
        self._draw_metrics(price_str, se_str, ci_str, bs_str)
        # Update greek strip
        self._draw_greeks(result["greeks"])
        # Update charts
        self._draw_histogram(result["payoffs"])
        self._draw_paths(result["paths"], K, T)
 
        self.txt_status.set_text(f"Done — {N:,} paths · {steps} steps · {style} {opt_type}")
        self.btn_run.label.set_text("Run Simulation")
        self.fig.canvas.draw_idle()
 
 
# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    app = OptionPricingApp()
    plt.show()