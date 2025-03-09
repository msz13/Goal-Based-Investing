using AdvancedPS

using Random
using Distributions
using Plots
using SSMProblems

Parameters = @NamedTuple begin
    a::Float64
    q::Float64
    r::Float64
end

mutable struct LinearSSM <: SSMProblems.AbstractStateSpaceModel
    X::Vector{Float64}
    observations::Vector{Float64}
    θ::Parameters
    LinearSSM(θ::Parameters) = new(Vector{Float64}(), θ)
    LinearSSM(y::Vector, θ::Parameters) = new(Vector{Float64}(), y, θ)
end

f(θ::Parameters, state, t) = Normal(θ.a * state, θ.q) # Transition density
g(θ::Parameters, state, t) = Normal(state, θ.r)         # Observation density
f₀(θ::Parameters) = Normal(0, θ.q^2 / (1 - θ.a^2)) 

SSMProblems.transition!!(rng::AbstractRNG, model::LinearSSM) = rand(rng, f₀(model.θ))

function SSMProblems.transition!!(
    rng::AbstractRNG, model::LinearSSM, state::Float64, step::Int
)
    return rand(rng, f(model.θ, state, step))
end

function SSMProblems.emission_logdensity(modeL::LinearSSM, state::Float64, step::Int)
    return logpdf(g(model.θ, state, step), model.observations[step])
end


function SSMProblems.transition_logdensity(
    model::LinearSSM, prev_state, current_state, step
)
    return logpdf(f(model.θ, prev_state, step), current_state)
end

AdvancedPS.isdone(::LinearSSM, step) = step > Tₘ


a = 0.9   # Scale
q = 0.32  # State variance
r = 1     # Observation variance
Tₘ = 200  # Number of observation
Nₚ = 20   # Number of particles
Nₛ = 500  # Number of samples
seed = 1  # Reproduce everything

θ₀ = Parameters((a, q, r))
rng = Random.MersenneTwister(seed)

x = zeros(Tₘ)
y = zeros(Tₘ)
x[1] = rand(rng, f₀(θ₀))
for t in 1:Tₘ
    if t < Tₘ
        x[t + 1] = rand(rng, f(θ₀, x[t], t))
    end
    y[t] = rand(rng, g(θ₀, x[t], t))
end

plot(x; label="x", xlabel="t")
plot!(y; seriestype=:scatter, label="y", xlabel="t", mc=:red, ms=2, ma=0.5)


model = LinearSSM(y, θ₀)
pgas = AdvancedPS.PGAS(Nₚ)
chains = sample(rng, model, pgas, Nₛ; progress=false);

particles = hcat([chain.trajectory.model.X for chain in chains]...)
mean_trajectory = mean(particles; dims=2);

scatter(particles; label=false, opacity=0.01, color=:black, xlabel="t", ylabel="state")
plot!(x; color=:darkorange, label="Original Trajectory")
plot!(mean_trajectory; color=:dodgerblue, label="Mean trajectory", opacity=0.9)

update_rate = sum(abs.(diff(particles; dims=2)) .> 0; dims=2) / Nₛ

plot(
    update_rate;
    label=false,
    ylim=[0, 1],
    legend=:bottomleft,
    xlabel="Iteration",
    ylabel="Update rate",
)
hline!([1 - 1 / Nₚ]; label="N: $(Nₚ)")