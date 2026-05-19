# Time Series Forecasting — WTI Crude Oil Prices

Two approaches to forecasting crude oil prices. One variable, then two.

The univariate part fits an ARIMA model to weekly WTI prices — unit root 
testing, ACF/PACF identification, model selection by AIC, and out-of-sample 
evaluation over a one-year holdout.

The VAR extends this by adding the Producer Price Index as a second variable. 
Granger causality runs in both directions, which justifies the system approach. 
The VAR forecast substantially outperforms the ARIMA benchmark  over the holdout period.

Full analysis in R. Impulse response functions, residual diagnostics, and 
forecast bands included in the report.

---

Completed with Cian Donlan — ECON42710 Advanced Econometrics, UCD.
