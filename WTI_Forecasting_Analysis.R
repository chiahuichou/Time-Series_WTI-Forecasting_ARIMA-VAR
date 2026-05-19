################################################################################
# ECON42710 Advanced Econometrics - Time Series Assignment
# Variable: DCOILWTICO - WTI Crude Oil Prices (student ID ends in 4)

#install.packages("urca")
#install.packages("vars")
#install.packages("stargazer")
################################################################################

library(urca)       # Augmented Dickey-Fuller unit root tests
library(vars)       # VAR estimation, IRF, Granger causality, forecasting
library(stargazer)  # Formatted regression output tables


################################################################################
########### QUESTION 1 - Univariate ARIMA Forecast
################################################################################

### Part (a) - Load data, plot, and assess stationarity visually ---------------

# Load the dataset
import <- read.csv("assignment.csv")

# Keep date and WTI price, remove missing values
oil <- import[, c("datestr", "DCOILWTICO")]
oil <- na.omit(oil)
oil$datestr <- as.Date(oil$datestr)

# Plot the raw series
plot(oil$datestr, oil$DCOILWTICO, type = "l",
     main = "WTI Crude Oil Prices",
     xlab = "Date",
     ylab = "Price (USD)",
     col  = "blue")

# The series does not appear stationary, since it shows persistent changes in
# level and no clear reversion to a constant mean. It looks more consistent
# with a stochastic trend / unit-root type process than with a deterministic
# trend-stationary process.


### Part (b) - Formal unit root test (ADF) ------------------------------------

# We use the Augmented Dickey-Fuller (ADF) test from the urca package.
# Two specifications are estimated:
#   "drift" - includes a constant but no deterministic time trend
#   "trend" - includes both a constant and a linear time trend
#
# The drift specification is the preferred one here because the graph suggests
# substantial persistence in the level of the series, but not a clear
# deterministic linear trend. A "none" specification would be inappropriate
# for an oil price series because it imposes no intercept.
#
# Since the data are quarterly, we allow up to 4 lagged difference terms,
# i.e. up to one year of quarterly dynamics, and let AIC select the lag length.

summary(ur.df(oil$DCOILWTICO, type = "drift", lags = 4, selectlags = "AIC"))
summary(ur.df(oil$DCOILWTICO, type = "trend", lags = 4, selectlags = "AIC"))

# Under the drift specification, the test statistic does not exceed the
# relevant critical value, so we fail to reject the null hypothesis of a
# unit root. Hence, DCOILWTICO appears non-stationary in levels, with
# evidence consistent with a unit root.


### Part (c) - ACF/PACF of differenced series; propose candidate models --------

# Because the level series contains evidence consistent with a unit root, we
# follow the Box-Jenkins approach and examine the ACF and PACF of the
# first-differenced series.

d_oil <- diff(oil$DCOILWTICO)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
acf(d_oil,  main = "ACF of differenced WTI price")
pacf(d_oil, main = "PACF of differenced WTI price")
par(mfrow = c(1, 1))

# Interpretation of correlograms:
#   - The ACF shows significant autocorrelations at the first 1-2 lags, then
#     cuts off sharply, consistent with a low-order MA process in differences.
#     In particular, MA(2) is plausible → candidate: ARIMA(0,1,2).
#   - The PACF also shows only low-order dynamics.
#     An ARMA(1,1) in differences is also plausible → candidate: ARIMA(1,1,1).
#
# Two candidate models:
#   Model 1: ARIMA(0,1,2)
#   Model 2: ARIMA(1,1,1)


### Part (d) - Estimate candidate ARIMA models --------------------------------

# Both models are estimated in levels with d = 1 imposed (one difference),
# consistent with the evidence of non-stationarity in part (b).

arima_012 <- arima(oil$DCOILWTICO, order = c(0, 1, 2))
arima_111 <- arima(oil$DCOILWTICO, order = c(1, 1, 1))

print(arima_012)
print(arima_111)

# ARIMA(0,1,2) estimates:
#   ma1 = 0.2573,  ma2 = -0.1705
#   sigma^2 = 65.86,  log-likelihood = -537.53,  AIC = 1081.06
#
# ARIMA(1,1,1) estimates:
#   ar1 = -0.2697,  ma1 = 0.5755
#   sigma^2 = 66.54,  log-likelihood = -538.30,  AIC = 1082.59


### Part (e) - Model selection: information criteria and residual diagnostics --

# --- Information criteria ---
AIC(arima_012, arima_111)
BIC(arima_012, arima_111)

# --- Residual ACF plots (visual check for remaining autocorrelation) ---
par(mfrow = c(1, 2))
acf(residuals(arima_012), main = "Residual ACF: ARIMA(0,1,2)")
acf(residuals(arima_111), main = "Residual ACF: ARIMA(1,1,1)")
par(mfrow = c(1, 1))

# --- Ljung-Box test for residual serial correlation ---
# Null hypothesis: residuals are white noise (no remaining autocorrelation)
# fitdf = number of estimated ARMA parameters
Box.test(residuals(arima_012), lag = 20, type = "Ljung", fitdf = 2)
Box.test(residuals(arima_111), lag = 20, type = "Ljung", fitdf = 2)

# Results:
#   ARIMA(0,1,2): AIC = 1081.06,  Ljung-Box p = 0.7153  (fail to reject H0)
#   ARIMA(1,1,1): AIC = 1082.59,  Ljung-Box p = 0.6719  (fail to reject H0)
#
# Both models pass the residual white-noise test, so both are statistically
# adequate. ARIMA(0,1,2) is selected as the preferred model because it has
# the lower AIC, slightly lower estimated variance, and higher log-likelihood.


### Part (f) - Generate forecasts for the final year --------------------------

# We use the preferred ARIMA(0,1,2) model.
# Strategy: estimate on all observations EXCEPT the last 4 quarters (the final
# year of quarterly data), then forecast forward 4 steps and compare with
# actuals on the same holdout window later used for the VAR comparison.

y <- oil$DCOILWTICO
dates <- oil$datestr
h <- 4             # hold-out horizon: 4 quarterly observations = 1 year
n <- length(y)

train <- y[1:(n - h)]
test  <- y[(n - h + 1):n]
test_dates  <- dates[(n - h + 1):n]

# Fit ARIMA(0,1,2) on training sample only
fit_012_forecast <- arima(train, order = c(0, 1, 2))

# Produce 4-step-ahead point forecasts and standard errors
fc    <- predict(fit_012_forecast, n.ahead = h)
pred  <- as.numeric(fc$pred)
lower <- pred - 1.96 * fc$se   # 95% lower forecast band
upper <- pred + 1.96 * fc$se   # 95% upper forecast band

# Save forecast plot with dates on the x-axis
png("1_f_quarterly.png", width = 1600, height = 900, res = 200)
plot(test_dates, test, type = "l", lwd = 2, col = "black",
     xlab = "Date", ylab = "WTI Price (USD)",
     main = "ARIMA(0,1,2): Quarterly Forecast for Final Year",
     ylim = range(c(test, pred, lower, upper), na.rm = TRUE))

lines(test_dates, pred,  col = "red",  lwd = 2)
lines(test_dates, lower, col = "blue", lty = 2)
lines(test_dates, upper, col = "blue", lty = 2)

legend("topright",
       legend = c("Actual", "Forecast", "95% forecast bands"),
       col    = c("black", "red", "blue"),
       lty    = c(1, 1, 2),
       lwd    = c(2, 2, 1),
       cex    = 0.85)
dev.off()

# The point forecast is approximately flat, consistent with an ARIMA(0,1,2)
# model without a drift term. Over a four-quarter horizon the forecast path is
# fairly smooth, while the intervals widen with the forecast horizon. Any sharp
# changes in oil prices over the final year are therefore unlikely to be tracked
# closely by this specification.


### Part (g) - Assess forecasting power ---------------------------------------

# Compute out-of-sample forecast errors
forecast_errors <- test - pred

# Standard forecast accuracy metrics
rmse_arima <- sqrt(mean(forecast_errors^2))  # Root Mean Squared Forecast Error
mae_arima  <- mean(abs(forecast_errors))      # Mean Absolute Error
me_arima   <- mean(forecast_errors)           # Mean Error (bias)

# Empirical coverage of the 95% forecast bands
coverage_arima <- mean(test >= lower & test <= upper)

# Print results
cat("--- ARIMA(0,1,2) Forecast Evaluation ---\n")
cat("RMSE     :", round(rmse_arima, 4), "\n")
cat("MAE      :", round(mae_arima,  4), "\n")
cat("Bias (ME):", round(me_arima,   4), "\n")
cat("95% Band Coverage:", round(coverage_arima * 100, 1), "%\n")

# On the corrected four-quarter holdout, the forecast errors are modest in
# magnitude for a crude-oil price series.
# Mean error is positive, so the model underpredicts oil prices on average over
# this final year.
# Coverage is 100%, but with only four holdout observations this mostly shows
# that the forecast bands are wide enough to contain the realised path.
# Overall: a reasonable quarterly benchmark, but inference is limited by the
# short evaluation window.


################################################################################
########### QUESTION 2 - VAR Forecast
################################################################################

### Part (a) - Choose a companion variable and justify ------------------------

# We choose PPIACO: Producer Price Index by Commodity - All Commodities (US).
#
# Economic justification:
# Crude oil is a primary production input across nearly all US industries.
# A rise in oil prices directly raises production costs, which feeds through
# into higher producer prices (PPI). Conversely, PPI movements reflect shifts
# in aggregate industrial demand, which affect the demand for oil and therefore
# its price. The oil price–PPI link is bidirectional and well-documented in the
# energy economics literature, making PPIACO a natural companion variable in a
# VAR system with WTI crude oil prices.

# Load the full dataset again, keeping both DCOILWTICO and PPIACO
import2 <- read.csv("assignment.csv")
df      <- import2[, c("datestr", "DCOILWTICO", "PPIACO")]
df$datestr <- as.Date(df$datestr)

# Keep only rows where BOTH series have observations
df <- na.omit(df)

cat("Q2 Sample period:", as.character(min(df$datestr)),
    "to", as.character(max(df$datestr)), "\n")
cat("Number of observations:", nrow(df), "\n")

# Plot both series for a visual overview
par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))

plot(df$datestr, df$DCOILWTICO, type = "l", col = "blue", lwd = 1.5,
     main = "WTI Crude Oil Price (DCOILWTICO)",
     xlab = "", ylab = "USD per barrel")

plot(df$datestr, df$PPIACO, type = "l", col = "darkred", lwd = 1.5,
     main = "Producer Price Index: All Commodities (PPIACO)",
     xlab = "Date", ylab = "Index (1982 = 100)")

par(mfrow = c(1, 1))


### Part (b) - Test stationarity of PPIACO ------------------------------------

# From Question 1, DCOILWTICO was treated as a non-stationary series and
# modelled with one difference. We now test PPIACO explicitly.
# Visual inspection shows a long upward trend with no clear mean-reversion,
# suggesting possible non-stationarity. We test with both drift and trend
# specifications.

cat("\n--- ADF test for PPIACO in levels (drift) ---\n")
summary(ur.df(df$PPIACO, type = "drift", lags = 4, selectlags = "AIC"))

cat("\n--- ADF test for PPIACO in levels (trend) ---\n")
summary(ur.df(df$PPIACO, type = "trend", lags = 4, selectlags = "AIC"))

# If we fail to reject H0 (unit root) in levels, test on first differences:
d_ppi_test <- diff(df$PPIACO)

cat("\n--- ADF test for delta(PPIACO) (drift) ---\n")
summary(ur.df(d_ppi_test, type = "drift", lags = 4, selectlags = "AIC"))

# If the level tests fail to reject a unit root and the differenced series is
# stationary, PPIACO can be treated as I(1). In that case, estimating the VAR
# in first differences is appropriate.
# (A full cointegration analysis is beyond the scope of this assignment.)


### Part (c) - Select lag length for the VAR ----------------------------------

# We work with first differences of both series since both appear to be I(1).
d_oil_var <- diff(df$DCOILWTICO)
d_ppi_var <- diff(df$PPIACO)

# Combine into a matrix (required format for the vars package)
var_data <- cbind(d_oil_var, d_ppi_var)
colnames(var_data) <- c("d_oil", "d_ppi")

# VARselect computes AIC, BIC (SC), and HQ for lag lengths 1 to 8
lag_select <- VARselect(var_data, lag.max = 8, type = "const")

cat("\n--- VAR Lag Selection Criteria ---\n")
print(lag_select$criteria)
cat("\nAIC selects:", lag_select$selection["AIC(n)"], "lags\n")
cat("BIC selects:", lag_select$selection["SC(n)"],  "lags\n")
cat("HQ  selects:", lag_select$selection["HQ(n)"],  "lags\n")

# We use the lag length chosen by AIC for the following reasons:
#   - AIC applies a lighter penalty for additional lags than BIC/SC, which
#     reduces the risk of under-fitting dynamic structure.
#   - For forecasting applications, AIC-selected models often outperform more
#     parsimonious BIC-selected models because under-fitting dynamics can be
#     costly for forecast accuracy.

p_chosen <- lag_select$selection["AIC(n)"]
cat("Chosen lag order p =", p_chosen, "\n")


### Part (d) - Estimate the VAR and report coefficients -----------------------

# Estimate VAR(p) in first differences with a constant included
var_model <- VAR(var_data, p = p_chosen, type = "const")

cat("\n--- VAR Model Summary ---\n")
summary(var_model)

# Extract the individual equation objects for formatted output
eq_oil <- var_model$varresult$d_oil
eq_ppi <- var_model$varresult$d_ppi

# Produce a formatted coefficient table using stargazer
# Change type = "text" to type = "latex" when compiling the PDF report
cat("\n--- Formatted coefficient table ---\n")
stargazer(eq_oil, eq_ppi,
          title      = "VAR Estimates: WTI Oil Price and PPI (first differences)",
          dep.var.labels = c("Delta Oil Price", "Delta PPI"),
          type       = "text")


### Part (e) - Impulse Response Functions (IRFs) ------------------------------

# Compute orthogonalised IRFs with 95% bootstrap confidence intervals.
# n.ahead = 12 quarters (3 years); runs = 500 bootstrap replications.
# Ordering: d_oil first, d_ppi second (Cholesky decomposition).

irf_oil_to_oil <- irf(var_model, impulse = "d_oil", response = "d_oil",
                      n.ahead = 12, boot = TRUE, runs = 500, ci = 0.95)

irf_oil_to_ppi <- irf(var_model, impulse = "d_oil", response = "d_ppi",
                      n.ahead = 12, boot = TRUE, runs = 500, ci = 0.95)

irf_ppi_to_oil <- irf(var_model, impulse = "d_ppi", response = "d_oil",
                      n.ahead = 12, boot = TRUE, runs = 500, ci = 0.95)

irf_ppi_to_ppi <- irf(var_model, impulse = "d_ppi", response = "d_ppi",
                      n.ahead = 12, boot = TRUE, runs = 500, ci = 0.95)

# Save each IRF separately at higher resolution for LaTeX
png("2_e_1.png", width = 1600, height = 1200, res = 200)
par(mar = c(4, 4, 1.2, 1), cex.axis = 1.1, cex.lab = 1.1)
plot(irf_oil_to_oil, main = "")
dev.off()

png("2_e_2.png", width = 1600, height = 1200, res = 200)
par(mar = c(4, 4, 1.2, 1), cex.axis = 1.1, cex.lab = 1.1)
plot(irf_oil_to_ppi, main = "")
dev.off()

png("2_e_3.png", width = 1600, height = 1200, res = 200)
par(mar = c(4, 4, 1.2, 1), cex.axis = 1.1, cex.lab = 1.1)
plot(irf_ppi_to_oil, main = "")
dev.off()

png("2_e_4.png", width = 1600, height = 1200, res = 200)
par(mar = c(4, 4, 1.2, 1), cex.axis = 1.1, cex.lab = 1.1)
plot(irf_ppi_to_ppi, main = "")
dev.off()

# Economic interpretation:
#
# Oil → PPI (top-right panel):
#   A positive oil price shock would typically be expected to raise PPI because
#   oil is a key production cost. A positive initial response that gradually
#   dies out would therefore be economically plausible.
#
# PPI → Oil (bottom-left panel):
#   A positive PPI shock, reflecting stronger aggregate demand or broader cost
#   pressures, may feed back into higher oil demand and therefore higher oil
#   prices. This effect would usually be expected to be smaller and less
#   persistent than the oil→PPI channel.
#
# Own-shock panels (diagonal):
#   Both own-shock responses should decay toward zero, confirming that
#   the estimated VAR is dynamically stable (all eigenvalues inside the
#   unit circle). We can verify this explicitly:
cat("\n--- VAR Stability Check ---\n")
print(roots(var_model))
# All roots should be < 1 in modulus for a stable VAR.


### Part (f) - Granger Causality Tests ----------------------------------------

# The Granger causality test asks: do past values of variable X contain
# statistically significant predictive information for variable Y, over and
# above what past values of Y alone already provide?
#
# Null hypothesis in each test: the "cause" variable does NOT Granger-cause
# the response variable (i.e. its lags are jointly zero in the equation for
# the response variable).

cat("\n--- Granger Causality: Does d_ppi Granger-cause d_oil? ---\n")
granger_ppi_to_oil <- causality(var_model, cause = "d_ppi")
print(granger_ppi_to_oil$Granger)

cat("\n--- Granger Causality: Does d_oil Granger-cause d_ppi? ---\n")
granger_oil_to_ppi <- causality(var_model, cause = "d_oil")
print(granger_oil_to_ppi$Granger)

# Interpretation guide:
#
# If d_ppi Granger-causes d_oil (p < 0.05):
#   Past PPI values contain useful predictive information for oil prices beyond
#   what lagged oil prices already tell us. Including PPIACO in the VAR is
#   therefore statistically justified, though it does not guarantee better
#   out-of-sample forecasts on every holdout sample.
#
# If d_oil Granger-causes d_ppi (p < 0.05):
#   Causation also runs from oil to PPI, consistent with the cost
#   pass-through mechanism (oil → production costs → producer prices).
#
# If neither causes the other:
#   The VAR may still improve forecasts through contemporaneous correlation
#   (captured in the covariance matrix of residuals), but the lagged
#   predictive content of each variable for the other is limited.


### Part (g) - VAR forecast of WTI crude oil price ----------------------------

# Hold out the same final 4 quarters used in Question 1
h_var <- 4

# Split differenced VAR data into training and test sets
n_diff <- nrow(var_data)
train_var <- var_data[1:(n_diff - h_var), , drop = FALSE]

# Choose lag length on TRAINING sample only
lag_select_train <- VARselect(train_var, lag.max = 8, type = "const")
p_train <- lag_select_train$selection["AIC(n)"]

# Re-estimate VAR on training sample
var_train <- VAR(train_var, p = p_train, type = "const")

# Forecast h_var steps ahead with 95% intervals
var_fc <- predict(var_train, n.ahead = h_var, ci = 0.95)

# Forecasts for oil-price CHANGES
fc_d_oil <- var_fc$fcst$d_oil[, "fcst"]
fc_lower <- var_fc$fcst$d_oil[, "lower"]
fc_upper <- var_fc$fcst$d_oil[, "upper"]

# Convert point forecasts back to LEVELS
last_level_train <- df$DCOILWTICO[nrow(df) - h_var]
pred_levels <- last_level_train + cumsum(fc_d_oil)

# Approximate level bands
lower_levels <- last_level_train + cumsum(fc_lower)
upper_levels <- last_level_train + cumsum(fc_upper)

# Actual holdout-period levels and dates
actual_levels <- df$DCOILWTICO[(nrow(df) - h_var + 1):nrow(df)]
actual_dates  <- df$datestr[(nrow(df) - h_var + 1):nrow(df)]

# Save forecast plot
png("2_g_quarterly.png", width = 1600, height = 900, res = 200)
plot(actual_dates, actual_levels, type = "l", lwd = 2, col = "black",
     ylim = range(c(actual_levels, pred_levels, lower_levels, upper_levels), na.rm = TRUE),
     xlab = "Date", ylab = "WTI Crude Oil Price (USD)",
     main = "VAR Forecast vs Actual: WTI Crude Oil Price (Final Year)")

lines(actual_dates, pred_levels,  col = "red",  lwd = 2)
lines(actual_dates, lower_levels, col = "blue", lwd = 1, lty = 2)
lines(actual_dates, upper_levels, col = "blue", lwd = 1, lty = 2)

legend("topright",
       legend = c("Actual", "VAR Forecast", "Approx. 95% Forecast Band"),
       col    = c("black", "red", "blue"),
       lty    = c(1, 1, 2),
       lwd    = c(2, 2, 1),
       cex    = 0.85)
dev.off()

### Part (h) - Compare VAR forecast vs ARIMA(0,1,2) from Question 1 ----------

# Compute VAR forecast accuracy metrics
var_errors   <- actual_levels - pred_levels
rmse_var     <- sqrt(mean(var_errors^2))
mae_var      <- mean(abs(var_errors))
me_var       <- mean(var_errors)
coverage_var <- mean(actual_levels >= lower_levels & actual_levels <= upper_levels)

cat("\n--- VAR Forecast Evaluation ---\n")
cat("RMSE     :", round(rmse_var, 4), "\n")
cat("MAE      :", round(mae_var,  4), "\n")
cat("Bias (ME):", round(me_var,   4), "\n")
cat("95% Band Coverage:", round(coverage_var * 100, 1), "%\n")

# ARIMA(0,1,2) results from Question 1 for comparison
# (these values come from running Part (g) of Question 1 above)
cat("\n--- Side-by-Side Comparison ---\n")
cat(sprintf("%-20s %10s %10s %12s\n", "Model", "RMSE", "MAE", "Bias"))
cat(sprintf("%-20s %10.4f %10.4f %12.4f\n", "ARIMA(0,1,2)", rmse_arima, mae_arima, me_arima))
cat(sprintf("%-20s %10.4f %10.4f %12.4f\n", "VAR",          rmse_var,   mae_var,   me_var))

# The ARIMA and VAR metrics are now computed on the same final four-quarter
# holdout window, so the RMSE, MAE, bias, and coverage measures are directly
# comparable on a like-for-like basis.
