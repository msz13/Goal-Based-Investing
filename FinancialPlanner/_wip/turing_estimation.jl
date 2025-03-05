using Turing, Distributions, Random, LinearAlgebra, StatsPlots
using Revise

includet("turing_model.jl")


T = 100  # Number of time steps
data, true_states = generate_data(T)


model = switching_model(data)

# Sampling
chain = sample(model, NUTS(), 1000)


y = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
];
N = length(y);
K = 3;

g = Gibbs(HMC(0.01, 50, :m, :T), PG(120, :s))
chn = sample(BayesHmm(y, 3), g, 1000);

summarize(get(chn, :s))

summarize(MCMCChains.group(chn, :m))

dist = filldist(Normal(0,1),2,2,2)

arraydist([Exponential(i) for i in 1:3])