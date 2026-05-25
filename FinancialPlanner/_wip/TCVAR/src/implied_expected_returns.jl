"""
    ImpliedExpectedReturns

Module for computing implied expected returns of equities using:
- A Vector Autoregressive (VAR) model to forecast extraordinary earnings growth
- A two-stage Dividend Discount Model (DDM) to link prices to expected returns
- Brent's root-finding algorithm (FLOWMath.jl) to solve for the implied return
"""
module ImpliedExpectedReturns

using FLOWMath: brent

export forecast_earnings_growth, calculate_price, solve_implied_return, find_expected_returns

# ──────────────────────────────────────────────────────────────────────────────
# Helper 1: Forecast earnings growth via a VAR(1) model
# ──────────────────────────────────────────────────────────────────────────────

"""
    forecast_earnings_growth(g0, var_params, n_extrao)

Forecast expected earnings growth for each year of the extraordinary-growth
period using a first-order Vector Autoregressive model.

# Arguments
- `g0::Vector{Float64}`: state vector at time t=0; the first element is the
  current earnings growth rate, remaining elements are other VAR variables
  (e.g. payout ratio, interest rate spread, …).
- `var_params::Matrix{Float64}`: VAR(1) companion matrix (K × K) such that
  the state evolves as  z_{t+1} = var_params * z_t  (intercept can be
  embedded via an augmented state).
- `n_extrao::Int`: number of years the extraordinary growth phase lasts.

# Returns
- `eg::Vector{Float64}`: length-`n_extrao` vector of expected earnings
  growth rates for years 1 … n_extrao.  Each entry is the first element
  of the forecasted state vector.
"""
function forecast_earnings_growth(
    g0::Vector{Float64},
    var_params::Matrix{Float64},
    n_extrao::Int64,
)
    K = length(g0)
    @assert size(var_params) == (K, K) "var_params must be $(K)×$(K)"

    eg = Vector{Float64}(undef, n_extrao)
    z = copy(g0)

    for t in 1:n_extrao
        z = var_params * z          # one-step-ahead forecast
        eg[t] = z[1]                # first element = earnings growth
    end

    return eg
end

# ──────────────────────────────────────────────────────────────────────────────
# Helper 2: Two-stage DDM price calculation
# ──────────────────────────────────────────────────────────────────────────────

"""
    calculate_price(eg, g_lt, r; payout=0.5, eps0=1.0)

Compute the fair price of a stock index using a two-stage Dividend Discount
Model given expected earnings growth rates and a discount rate.

## Two-stage DDM

    V = Σ_{t=1}^{T} D_t / (1+r)^t   +   D_{T+1} / [(r − g_lt)(1+r)^T]

where
- T           = length(eg), the extraordinary growth period
- D_t         = payout × E_t, with  E_t = E_{t-1} × (1 + eg[t])
- g_lt        = long-run sustainable growth rate (terminal)
- D_{T+1}     = payout × E_T × (1 + g_lt)

# Arguments
- `eg::Vector{Float64}`: expected earnings growth rates during the
  extraordinary period (output of `forecast_earnings_growth`).
- `g_lt::Float64`: long-term sustainable earnings growth rate.
- `r::Float64`: discount rate (expected return) used to discount cash flows.
- `payout::Float64=0.5`: constant dividend payout ratio (D/E).
- `eps0::Float64=1.0`: current (time-0) earnings per share / index earnings.

# Returns
- `price::Float64`: model-implied fair price of the stock index.
"""
function calculate_price(
    eg::Vector{Float64},
    g_lt::Float64,
    r::Float64;
    payout::Float64 = 0.5,
    eps0::Float64 = 1.0,
)
    T = length(eg)

    # ---- Stage 1: extraordinary growth period ----
    E = eps0
    pv_stage1 = 0.0
    for t in 1:T
        E *= (1.0 + eg[t] + g_lt)
        D = payout * E
        pv_stage1 += D / (1.0 + r)^t
    end

    # ---- Stage 2: Gordon-growth terminal value ----
    E_terminal = E * (1.0 + g_lt)
    D_terminal = payout * E_terminal
    terminal_value = D_terminal / (r - g_lt)   # Gordon growth model
    pv_stage2 = terminal_value / (1.0 + r)^T

    return pv_stage1 + pv_stage2
end

# ──────────────────────────────────────────────────────────────────────────────
# Helper 3: Solve for implied expected return via Brent
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_implied_return(eg, g_lt, price; payout=0.5, eps0=1.0,
                         r_lo=1e-6, r_hi=1.0)

Find the implied expected return that equates the two-stage DDM price
(from `calculate_price`) to the observed market `price`, using Brent's
root-finding method from FLOWMath.jl.

Solves  calculate_price(eg, g_lt, r; payout, eps0) − price = 0  for r.

# Arguments
- `eg::Vector{Float64}`: expected earnings growth rates during the
  extraordinary period (output of `forecast_earnings_growth`).
- `g_lt::Float64`: long-term sustainable earnings growth rate.
- `price::Float64`: observed stock index price.
- `payout::Float64=0.5`: constant dividend payout ratio (D/E).
- `eps0::Float64=1.0`: current (time-0) earnings per share / index earnings.
- `r_lo::Float64=1e-6`: lower bracket for the search (must give V > price).
- `r_hi::Float64=1.0`: upper bracket for the search (must give V < price).

# Returns
- `r_star::Float64`: implied expected return.
"""
function solve_implied_return(
    eg::Vector{Float64},
    g_lt::Float64,
    price::Float64;
    payout::Float64 = 0.5,
    eps0::Float64 = 1.0,
    r_lo::Float64 = 1e-6,
    r_hi::Float64 = 1.0,
)
    residual(r) = calculate_price(eg, g_lt, r; payout=payout, eps0=eps0) - price

    r_star, info = brent(residual, r_lo, r_hi)
   

    if info.flag != "CONVERGED"
        @warn "Brent solver did not converge" info
    end

    return r_star
end

# ──────────────────────────────────────────────────────────────────────────────
# Main: find implied expected returns for a time series
# ──────────────────────────────────────────────────────────────────────────────

"""
    find_expected_returns(earnings_growth, var_params, g_lt, prices,
                          n_extrao; payout=0.5, eps0=ones(T),
                          r_lo=1e-6, r_hi=1.0)

Compute the implied expected return of equities for each observation in the
sample by combining the VAR earnings-growth forecast with the two-stage DDM.

# Arguments
- `earnings_growth::Vector{Vector{Float64}}`: for each time t, the VAR state
  vector whose first element is the observed earnings growth rate.
  Length = number of observations.
- `var_params::Matrix{Float64}`: VAR(1) companion matrix (K × K).
- `g_lt::Vector{Float64}`: long-term expected growth rate for each t.
- `prices::Vector{Float64}`: observed stock index price at each t.
- `n_extrao::Int`: number of years of extraordinary growth.

# Keyword Arguments
- `payout::Float64=0.5`: constant dividend payout ratio.
- `eps0::Vector{Float64}`: base earnings level at each t
  (defaults to a vector of ones).
- `r_lo::Float64=1e-6`: lower bracket for Brent search.
- `r_hi::Float64=1.0`: upper bracket for Brent search.

# Returns
- `implied_returns::Vector{Float64}`: implied expected return at each t.
"""
function find_expected_returns(
    earnings_growth::Matrix{Float64},
    var_params::Matrix{Float64},
    g_lt::Vector{Float64},
    prices::Vector{Float64},
    n_extrao::Int64;
    payout::Float64 = 0.5,
    eps0::Vector{Float64} = ones(length(prices)),
    r_lo::Float64 = 1e-6,
    r_hi::Float64 = 1.0,
)
    T = length(prices)
    @assert size(earnings_growth, 1) == T "earnings_growth must have $T entries"
    @assert length(g_lt) == T "g_lt must have $T entries"

    implied_returns = Vector{Float64}(undef, T)

    for t in 1:T
        # Step 1: forecast extraordinary-period earnings growth via VAR
        eg = forecast_earnings_growth(earnings_growth[t, :], var_params, n_extrao)
        eg = eg 

        # Step 2: solve for the implied return with two-stage DDM + Brent
        implied_returns[t] = solve_implied_return(
            eg, g_lt[t], prices[t];
            payout = payout,
            eps0 = eps0[t],
            r_lo = r_lo,
            r_hi = r_hi,
        )
    end

    return implied_returns
end

end # module
