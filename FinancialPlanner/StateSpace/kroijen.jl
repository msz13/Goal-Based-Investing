
include("kroijen_model.jl")

using Statistics

ρ = 0.968
pd_hat = log(ρ/(1-ρ))
κ = log(1+exp(pd_hat)) - ρ*pd_hat

A = 3.612 
B1 = 13.484
B2 = 2.616

γ0 = 0.060 
γ1 =  0.638
δ0 = 0.086
δ1 = 0.957

σD = 0.089 
σg = 0.077
σu = 0.016
ρuD = -0.344
ρuG = 0.805
ρgD = 0.0

ρM = 0.586
σM = 0.054 

A_ = κ/(1-ρ) + (γ0 - δ0)/(1-ρ)
B1_ = 1/(1-ρ*δ1)
B2_ = 1/(1-ρ*γ1)



initial_obs = [.0629, 3.4]
initial_state = [.06, 0, 0, 0, 0, 0]

model = KroijenModel(ρ, A, B1, B2, γ0, γ1, δ1, σD, σg, σu, ρuD, ρuG, ρgD, ρM, σM)


n_steps = 30
n_scenarios = 2000


state = zeros(n_scenarios, n_steps, 6)
obs = zeros(n_scenarios, n_steps, 2)

for s in 1:n_scenarios
    state[s,1,:] = sim_state_step(model, initial_state)
    obs[s, 1,:] = sim_obs_step(model, state[s,1,:], initial_obs)
    for t in 2:n_steps
        state[s,t,:] = sim_state_step(model,  state[s,t-1,:])
        obs[s,t,:] = sim_obs_step(model, state[s,t,:], obs[s,t-1,:])
    end 
end


quantile(obs[:,2,1], [0.05, .25, .5, .75, .95])
quantile(obs[:,2,2], [0.05, .25, .5, .75, .95])

returns = κ .+ ρ*obs[:,3,2] + obs[:,3,1] - obs[:,2,2] 

quantile(exp.(returns), [0.02, .25, .5, .75, .98])

ret = κ + ρ*3.97 + .1085 - 4.04


create_M0(γ0,δ1,A)


s_cov = create_state_cov(ρ, B1, B2, σD, σg, σu, ρuD, ρuG, ρgD, ρM, σM)

sqrt.(diag(s_cov))


sm1 = zeros(6)

model.M0 + model.M1 * initial_obs + model.M2 * sm1


sm2 = cholesky(s_cov).L * fill(-3.5,4)

model.M0 + model.M1 * initial_obs + model.M2 * model. R * sm2

sm3 = [.38, .27, -0.078, 0.073, 0.109, -0.109]

obs3 = model.M0 + model.M1 * [0., 2.69] + model.M2 * sm3

sm4 = [-.38, -.27, 0.078, -0.073, -0.109, 0.109]

obs4 = model.M0 + model.M1 * [0., 2.69] + model.M2 * sm4

returns = κ .+ ρ*obs4[2] + obs4[1] - obs3[2]
exp(returns) 


using Distributions

r = -.25

cdf(Normal(.08, .16), r)

cdf(LocationScale(.08, .16, TDist(100)), r)