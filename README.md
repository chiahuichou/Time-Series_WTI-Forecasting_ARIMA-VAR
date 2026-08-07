Two approaches to forecasting crude oil prices. One variable, then two.

The univariate part fits an ARIMA model to quarterly WTI prices — unit root testing, ACF/PACF identification, model selection by AIC, and out-of-sample evaluation over a four-quarter holdout.

The VAR extends this by adding the Producer Price Index as a second variable. Granger causality runs in both directions, which statistically justifies the system approach. However, on the same holdout period, the ARIMA benchmark outperforms the VAR in point-forecast accuracy, illustrating that statistically significant predictive linkages do not automatically translate into better out-of-sample forecasts.

Full analysis in R. Impulse response functions, residual diagnostics, and forecast bands included in the report.

Completed with Cian Donlan — ECON42710 Advanced Econometrics, UCD.
