import Pkg
pwd()
Pkg.activate("./FinancialPlanner")

using TimeSeries, XLSX
using Statistics
using LinearAlgebra
using Dates
using Random
using DataFrames
using Plots

"""
Calculate annualized realized volatility over a given window
Assumes daily returns and 252 trading days per year
"""
function realized_volatility(returns::Vector{Float64}, window::Int; 
                            trading_days_per_year::Int=252)
    n = length(returns)
    rv = Float64[]
    
    for i in window:n
        window_returns = returns[(i-window+1):i]
        # Sum of squared returns over window
        sum_sq_returns = sum(window_returns.^2)
        # Annualized volatility
        vol = sqrt(sum_sq_returns * trading_days_per_year / window)
        push!(rv, vol)
    end
    
    return rv
end


"""
Fit AR(1) model: y_t = φ₀ + φ₁*y_{t-1} + ε_t
Returns coefficients (intercept, AR coefficient) and residuals
"""
function fit_ar1(data::Vector{Float64})
    n = length(data)
    
    # Prepare lagged data
    y = data[2:end]  # y_t
    y_lag = data[1:(end-1)]  # y_{t-1}
    
    # Add intercept column
    X = hcat(ones(length(y_lag)), y_lag)
    
    # OLS estimation: β = (X'X)^(-1)X'y
    β = (X' * X) \ (X' * y)
    
    # Calculate residuals
    y_pred = X * β
    residuals = y - y_pred
    
    # Calculate statistics
    sse = sum(residuals.^2)
    mse = sse / (length(y) - 2)
    se = sqrt.(diag(inv(X' * X) * mse))
    
    return (
        intercept = β[1],
        ar_coef = β[2],
        residuals = residuals,
        std_errors = se,
        fitted = y_pred
    )
end

"""
Forecast h-steps ahead using AR(1) model
"""
function forecast_ar1(intercept::Float64, ar_coef::Float64, 
                      last_value::Float64, horizon::Int)
    forecasts = Float64[]
    current = last_value
    
    for h in 1:horizon
        next_val = intercept + ar_coef * current
        push!(forecasts, next_val)
        current = next_val
    end
    
    return forecasts
end

"""
Generate one-step-ahead predictions for all data points
Returns predictions aligned with the original data (first value is NaN)
"""
function predict_ar1_insample(intercept::Float64, ar_coef::Float64, 
                               data::Vector{Float64})
    n = length(data)
    predictions = Vector{Float64}(undef, n)
    predictions[1] = NaN  # No prediction for first observation
    
    # One-step-ahead predictions: ŷ_t = φ₀ + φ₁*y_{t-1}
    for t in 2:n
        predictions[t] = intercept + ar_coef * data[t-1]
    end
    
    return predictions
end

"""
Calculate prediction errors (actual - predicted)
"""
function prediction_errors(data::Vector{Float64}, predictions::Vector{Float64})
    errors = data .- predictions
    # First value will be NaN
    return errors
end



# Example usage with simulated data
println("=== Equity Risk Premium via Realized Volatility AR(1) ===\n")


# Read CSV file with price data
# Expected format: Date, Price columns (or Date, Open, High, Low, Close)
df = DataFrame(XLSX.readtable("FinancialPlanner/data/spx_daily.xlsx", "SPX", infer_eltypes=false))

data_source = TimeArray(df; timestamp = :Data)


println("Loaded $(length(data_source)) observations")
println("Date range: $(timestamp(data_source)[1]) to $(timestamp(data_source)[end])")
println()

# Extract prices (assumes column named "Close" or first data column)
# Adjust column name as needed: :Close, :Price, :AdjClose, etc.

# Calculate log returns
log_returns = diff(log.(data_source[:SPX]))
log_returns



# Calculate 2-month (≈42 trading days) realized volatility
window = 42
rv_data = realized_volatility(values(log_returns), window)

# Get corresponding dates for realized volatility
# Start from window+1 since we lose window observations
rv_dates = timestamp(data_source)[(window+1):end]

println("Realized Volatility Statistics:")
println("  Mean: ", round(mean(rv_data), digits=4))
println("  Std:  ", round(std(rv_data), digits=4))
println("  Min:  ", round(minimum(rv_data), digits=4))
println("  Max:  ", round(maximum(rv_data), digits=4))
println()

# Fit AR(1) model
model = fit_ar1(rv_data)

println("AR(1) Model Results:")
println("  y_t = φ₀ + φ₁*y_{t-1} + ε_t")
println()
println("  Intercept (φ₀): ", round(model.intercept, digits=4), 
        " (SE: ", round(model.std_errors[1], digits=4), ")")
println("  AR Coef (φ₁):   ", round(model.ar_coef, digits=4),
        " (SE: ", round(model.std_errors[2], digits=4), ")")
println("  Residual Std:   ", round(std(model.residuals), digits=4))
println()

# Check stationarity
if abs(model.ar_coef) < 1
    println("✓ Model is stationary (|φ₁| < 1)")
else
    println("⚠ Warning: Model may be non-stationary (|φ₁| ≥ 1)")
end
println()

# Unconditional mean (long-run average volatility)
unconditional_mean = model.intercept / (1 - model.ar_coef)
println("Unconditional mean volatility: ", round(unconditional_mean, digits=4))
println()

# Generate in-sample predictions for all data points
predictions = predict_ar1_insample(model.intercept, model.ar_coef, rv_data)
errors = prediction_errors(rv_data, predictions)


# Create TimeArray with predictions
predicted_ts = TimeArray(rv_dates, predictions, [:PredictedVol])
error_ts = TimeArray(rv_dates, errors, [:PredictionError])

println("In-Sample Prediction Statistics:")
println("  Mean Absolute Error:  ", round(mean(abs.(skipmissing(errors))), digits=4))
println("  Root Mean Squared Error: ", round(sqrt(mean(skipmissing(errors).^2)), digits=4))
println("  Mean Error (bias):    ", round(mean(skipmissing(errors)), digits=4))
println()

# Show sample of predictions vs actual
println("Sample: Actual vs Predicted (last 10 observations):")
for i in (length(rv_data)-9):length(rv_data)
    if i > 1  # Skip first observation
        println("  t=$i: Actual=$(round(rv_data[i], digits=4)), ",
                "Predicted=$(round(predictions[i], digits=4)), ",
                "Error=$(round(errors[i], digits=4))")
    end
end
println()

# Plot realized vs predicted volatility
plot_dates = rv_dates[2:end]  # Skip first date (NaN prediction)
plot_actual = rv_data[2:end]
plot_predicted = predictions[2:end]

p1 = plot(plot_dates, plot_actual, 
         label="Realized Volatility",
         linewidth=2,
         color=:blue,
         xlabel="Date",
         ylabel="Annualized Volatility",
         title="Realized vs Predicted Volatility (AR(1) Model)",
         legend=:topright,
         size=(1000, 600))

plot!(p1, plot_dates, plot_predicted,
      label="Predicted Volatility",
      linewidth=2,
      color=:red,
      linestyle=:dash)

display(p1)

#= # Plot prediction errors
p2 = plot(plot_dates, errors[2:end],
         label="Prediction Error",
         linewidth=1.5,
         color=:green,
         xlabel="Date",
         ylabel="Error (Actual - Predicted)",
         title="AR(1) Model Prediction Errors",
         legend=:topright,
         size=(1000, 400))

hline!(p2, [0], color=:black, linestyle=:dash, label="Zero Line", linewidth=1)

display(p2) =#