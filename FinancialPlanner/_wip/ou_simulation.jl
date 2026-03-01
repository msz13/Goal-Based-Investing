import Pkg
Pkg.activate("./FinancialPlanner")

using Random, Statistics, Plots, StatsBase, Printf

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
Random.seed!(42)

N     = 25      # time steps
dt    = 1.0   # step size
n_sim = 5000       # number of Monte Carlo paths

# Standard OU parameters
θ  = .8          # mean-reversion speed
μ  = 2.0          # long-run mean
σ  = 1.0          # volatility

# Stochastic trend parameters
σ_trend = 0.1    # diffusion of the trend

X0 = 2.0 # initial level of inflation

# ─────────────────────────────────────────────
# 1. Standard Ornstein–Uhlenbeck
#    dX = θ(μ - X)dt + σ dW
# ─────────────────────────────────────────────
function simulate_ou(n_steps, dt, θ, μ, σ; x0=0.0)
    x = zeros(n_steps + 1)
    x[1] = x0
    for t in 1:n_steps
        dW    = sqrt(dt) * randn()
        x[t+1] = μ + θ * (μ - x[t]) * dt + σ * dW
    end
    return x
end

# ─────────────────────────────────────────────
# 2. OU with stochastic trend (mean is a random walk)
#    dμ_t = σ_trend dW2
#    dX   = θ(μ_t - X)dt + σ dW1
# ─────────────────────────────────────────────
function simulate_ou_stochastic_trend(n_steps, dt, θ, μ0, σ, σ_trend; x0=0.0)
    x  = zeros(n_steps + 1)
    μt = zeros(n_steps + 1)
    x[1]  = x0
    μt[1] = μ0
    for t in 1:n_steps
        dW1   = sqrt(dt) * randn()
        dW2   = sqrt(dt) * randn()
        μt[t+1] = μt[t] + σ_trend * dW2
        x[t+1]  = μt[t] + θ * (μt[t] - x[t]) * dt + σ * dW1
    end
    return x, μt
end

# ─────────────────────────────────────────────
# Run Monte Carlo simulations
# ─────────────────────────────────────────────
paths_ou    = [simulate_ou(N, dt, θ, μ, σ; x0=X0) for _ in 1:n_sim]
paths_ou_st = [simulate_ou_stochastic_trend(N, dt, θ, μ, σ, σ_trend; x0=X0)[1] for _ in 1:n_sim]

# Convert to matrices (n_sim × n_steps+1)
mat_ou    = reduce(hcat, paths_ou)'
mat_ou_st = reduce(hcat, paths_ou_st)'

# ─────────────────────────────────────────────
# Quantile comparison across time
# ─────────────────────────────────────────────
qs = [0.05, 0.25, 0.50, 0.75, 0.95]
t_grid = (0:N) .* dt

function quantile_bands(mat, qs)
    [map(t -> quantile(mat[:, t], q), 1:size(mat, 2)) for q in qs]
end

bands_ou    = quantile_bands(mat_ou, qs)
bands_ou_st = quantile_bands(mat_ou_st, qs)

#= # ─────────────────────────────────────────────
# Plot 1 – Sample paths
# ─────────────────────────────────────────────
p1 = plot(title="Standard OU – Sample Paths (10 shown)",
          xlabel="Time", ylabel="X(t)", legend=false, size=(800, 400))
for i in 1:10
    plot!(p1, t_grid, paths_ou[i], alpha=0.6, lw=0.8)
end

p2 = plot(title="OU with Stochastic Trend – Sample Paths (10 shown)",
          xlabel="Time", ylabel="X(t)", legend=false, size=(800, 400))
for i in 1:10
    x, μt = simulate_ou_stochastic_trend(N, dt, θ, μ, σ, σ_trend)
    plot!(p2, t_grid, x,  alpha=0.6, lw=0.8)
    plot!(p2, t_grid, μt, alpha=0.3, lw=0.8, ls=:dash)
end

#paths_plot = plot(p1, "./FinancialPlanner/_wip/outputs/ou_paths.png")

# ─────────────────────────────────────────────
# Plot 2 – Quantile fan charts (side by side)
# ─────────────────────────────────────────────
colors = [:blue, :green, :red, :green, :blue]
alphas = [0.5,   0.6,   0.9,   0.6,   0.5  ]
labels = ["5%", "25%", "50%", "75%", "95%"]

function fan_plot(bands, title_str)
    p = plot(title=title_str, xlabel="Time", ylabel="X(t)", size=(700, 400))
    for (i, b) in enumerate(bands)
        plot!(p, t_grid, b, label=labels[i],
              color=colors[i], alpha=alphas[i], lw= i==3 ? 2 : 1,
              ls = i==3 ? :solid : :dash)
    end
    # shaded inter-quartile region
    plot!(p, t_grid, bands[2], fillrange=bands[4],
          fillalpha=0.15, color=:green, label="IQR", lw=0)
    return p
end

q1 = fan_plot(bands_ou,    "Standard OU – Quantile Fan")
q2 = fan_plot(bands_ou_st, "OU + Stochastic Trend – Quantile Fan")
fan_plot_fig = plot(q1, q2, layout=(1, 2), size=(1200, 450))
savefig(fan_plot_fig, "./FinancialPlanner/_wip/outputs/ou_quantile_fans.png")

# ─────────────────────────────────────────────
# Plot 3 – IQR width comparison (spread over time)
# ─────────────────────────────────────────────
iqr_ou    = bands_ou[4]    .- bands_ou[2]
iqr_ou_st = bands_ou_st[4] .- bands_ou_st[2]

spread_plot = plot(t_grid, iqr_ou,    label="Standard OU",        lw=2, color=:blue)
plot!(t_grid, iqr_ou_st, label="OU + Stochastic Trend", lw=2, color=:red,
      title="IQR Width Over Time (uncertainty comparison)",
      xlabel="Time", ylabel="Q75 – Q25", size=(900, 400))
savefig(spread_plot, "./FinancialPlanner/_wip/outputs/ou_iqr_comparison.png") =#

# ─────────────────────────────────────────────
# Terminal distribution statistics
# ─────────────────────────────────────────────
println("=== Terminal Distribution Statistics (t = $(N*dt)) ===")
println("\n  Standard OU:")
for (q, b) in zip(qs, bands_ou)
    @printf("    Q%.0f%%  = %+.4f\n", q*100, b[10])
end
println("\n  OU + Stochastic Trend:")
for (q, b) in zip(qs, bands_ou_st)
    @printf("    Q%.0f%%  = %+.4f\n", q*100, b[10])
end
#= println("\n  Spread ratio (IQR_trend / IQR_standard) at terminal time: ",
        round(iqr_ou_st[end] / iqr_ou[end], digits=3)) =#

println("\nAll figures saved to /mnt/user-data/outputs/")

