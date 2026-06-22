# Monte Carlo Option Pricing Engine

A desktop GUI application that prices European and American stock options using Monte Carlo simulation, with analytical Black-Scholes benchmarking and live Greeks display.

---

## What It Does

You type in a few numbers — stock price, strike, volatility, time to expiry — hit **Run Simulation**, and the app:

1. Simulates thousands of random price paths for the underlying stock under the risk-neutral measure (Geometric Brownian Motion)
2. Computes the expected discounted payoff across all those paths — that's your option price
3. Shows you a confidence interval so you know how much to trust the number
4. Displays the analytical Black-Scholes price alongside for comparison
5. Renders the Greeks (Δ Γ Θ ν ρ) from closed-form formulas
6. Draws two charts: a histogram of simulated payoffs and a sample of 20 price paths

---

## Requirements

```
Python 3.9+
numpy
scipy
matplotlib
```

Install everything at once:

```bash
pip install numpy scipy matplotlib
```

Matplotlib needs a GUI backend to open a window. If you're on:
- **Windows / Linux with Tkinter** → works out of the box (TkAgg backend)
- **macOS** → works out of the box (MacOSX backend)
- **Linux without Tkinter** → install it: `sudo apt install python3-tk`
- **Headless server** → you'll need to forward a display or use a virtual framebuffer

---

## Running It

```bash
python3 monte_carlo_options.py
```

The window opens and runs an initial simulation automatically with the default parameters.

---

## The Interface

### Left Panel — Controls

| Control | What to enter |
|---|---|
| **Option Type** | Call or Put |
| **Exercise Style** | European (exercise at expiry only) or American (exercise any time) |
| **Spot (S)** | Current stock price, e.g. `100` |
| **Strike (K)** | The price at which you can buy/sell the stock, e.g. `105` |
| **Rate % (r)** | Annual risk-free interest rate as a percentage, e.g. `5` for 5% |
| **Vol % (σ)** | Implied volatility as a percentage, e.g. `20` for 20% |
| **Expiry yrs (T)** | Time to expiration in years, e.g. `1.0` for one year, `0.25` for three months |
| **Div yield % (q)** | Continuous dividend yield as a percentage — use `0` if none |
| **Simulations** | Number of random paths to generate. `20000` is a good default; more = slower but tighter |
| **Time steps** | How many steps per path. `252` mirrors daily trading days in a year |
| **Conf. level %** | Width of the confidence interval, e.g. `95` for a 95% CI |

After adjusting any values, press **Run Simulation**.

### Right Panel — Results

**Metric Strip (top row)**

| Box | Meaning |
|---|---|
| **MC Price** | The simulated option price — your main output |
| **Std Error** | Standard error of the MC estimate; smaller is better (more paths → smaller) |
| **Conf. Interval** | The range you can be 95% (or whatever you set) confident the true price falls in |
| **BS Price** | Analytical Black-Scholes price for comparison (shown for European options only) |

**Greeks Strip (second row)**

Greeks measure how the option price changes when one input moves while everything else stays fixed. Values are displayed in market-standard units:

| Greek | What it measures | Units |
|---|---|---|
| **Δ Delta** | Price sensitivity to a $1 move in the stock | $/$ |
| **Γ Gamma** | Rate of change of Delta per $1 move in the stock | per $ |
| **Θ Theta** | Price decay per calendar day | $/day |
| **ν Vega** | Price change per 1 volatility point move (e.g. 20% → 21%) | per vol point |
| **ρ Rho** | Price change per 1 interest rate point move (e.g. 5% → 6%) | per rate point |

Positive values are shown in green; negative in red.

**Payoff Distribution Chart**

A histogram of the discounted payoffs from all simulated paths. The dark bar on the far left is the mass of paths that expired out of the money (zero payoff). The colored bars to the right show the distribution of profitable outcomes. A wide spread means the option price is uncertain; a tight cluster means the estimate is stable.

**Sample Price Paths Chart**

Twenty of the simulated stock price paths from today to expiry, with the strike price overlaid as a dashed red line. Paths above the line (for a call) are the ones that pay off. This is purely illustrative — the pricing uses all N paths, not just these 20.

---

## How the Pricing Works

### European Options

The simplest case. All that matters is where the stock ends up at expiry.

1. Simulate N stock prices at time T by drawing random log-returns from a normal distribution
2. Compute the payoff for each path: `max(S_T − K, 0)` for a call, `max(K − S_T, 0)` for a put
3. Discount all payoffs back to today using `e^{−rT}`
4. Average them — that's the price

**Variance reduction:** The app uses *antithetic variates* — for every random draw `z`, it also uses `−z`. This pairs paths that tend to cancel each other's noise, roughly halving the variance for free.

### American Options (Longstaff-Schwartz)

American options can be exercised at any point before expiry, which makes them harder to price — you need to decide at each time step whether exercising now beats waiting.

The app uses the **Longstaff-Schwartz least-squares Monte Carlo** method:

1. Simulate full paths from today to expiry (same GBM as above)
2. Start at expiry with the terminal payoff
3. Walk backward through time one step at a time. At each step, for paths that are currently in the money, fit a polynomial regression of future discounted cashflows against the current stock price
4. The regression predicts the "continuation value" — how much you'd expect to get by waiting
5. If the immediate payoff beats the continuation value, exercise early; otherwise keep waiting
6. Roll the cashflows back to today

This typically gives American prices slightly above their European counterparts for puts (early exercise can be valuable), and equal or close for calls on non-dividend-paying stocks (early exercise of calls is rarely optimal without dividends).

### Greeks

All Greeks are computed analytically from the Black-Scholes closed-form formulas, not estimated from the simulation. This means they are exact (given the model assumptions) and don't carry simulation noise.

---

## Practical Tips

**Tightening the confidence interval:** Double the number of simulations. The standard error scales as `1/√N`, so going from 5,000 to 20,000 paths cuts the error in half.

**Speed vs. accuracy tradeoff for Americans:** The Longstaff-Schwartz method is slower than European pricing because it does a regression at every time step. Reducing `Time steps` to `50` or `100` speeds things up considerably with modest accuracy loss.

**Checking your inputs:** If the MC price and BS price diverge significantly for a European option, something may be off — check that your volatility and rate are entered as percentages (e.g. `20`, not `0.20`).

**Out-of-the-money options:** When the strike is far from the spot, most paths expire worthless and the histogram will be dominated by the zero-payoff bar. This is normal — it reflects the actual distribution. The price will be small and the relative standard error will be large; increase simulations to compensate.

---

## File Structure

```
monte_carlo_options.py    # Everything — models, pricing, and GUI in one file
README.md                 # This file
```

The entire application is self-contained in a single file. There are no configuration files, databases, or network calls.

---

## Limitations and Assumptions

- **Constant volatility:** The model uses a single fixed volatility σ for the entire simulation. Real markets have volatility smiles and term structure — this model ignores them.
- **Constant interest rate:** The risk-free rate r is fixed. In practice rates change over time.
- **Continuous dividends:** The dividend yield q is modeled as a continuous stream. If the stock pays discrete quarterly dividends, this is an approximation.
- **No transaction costs or bid-ask spread:** The model gives a theoretical mid-market price.
- **Log-normal stock price:** GBM assumes stock returns are normally distributed. Real returns have fat tails and skew.

For educational and research use. Not financial advice.
