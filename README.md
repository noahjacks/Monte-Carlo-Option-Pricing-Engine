# Monte Carlo Option Pricing Engine

An interactive option pricing tool that runs entirely in the browser.
No server or installation required — powered by PyScript (Python + WebAssembly).

## Features
- European and American options (calls and puts)
- Geometric Brownian Motion with antithetic variates
- Longstaff-Schwartz (LSM) for American early exercise
- Black-Scholes price and Greeks for comparison
- Payoff distribution histogram and sample path chart

## How it works
The pricing engine is written in pure Python using NumPy and SciPy.
PyScript compiles these to WebAssembly so they run directly in the browser.
