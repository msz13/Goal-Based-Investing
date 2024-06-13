using Clustering
using StatsBase


function cluster_moments(data, n_clusters)
    dt = fit(ZScoreTransform, data, dims=2)
    standarized = StatsBase.transform(dt,data)

    clusters = kmeans(standarized,n_clusters)
    c = assignments(clusters)

    cluster_means = zeros(2,n_clusters)
    cluster_cov = zeros(2,2,n_clusters)

    for cluster in 1:n_clusters
        cluster_means[:,cluster] = mean(period_1[:,c .== cluster],dims=2) 
        cluster_cov[:,:,cluster] = cov(period_1[:,c .== cluster],dims=2) 
    end

    return  cluster_means, cluster_cov 
end

function guess_init(n_scenarios)
    regimes_probs = rand(1:100,n_scenarios)
    return regimes_probs/sum(regimes_probs)
end

function guess_tmatrix(n_scenarios)
    regimes_probs = rand(1:100,n_scenarios,n_scenarios)
    return regimes_probs ./ sum(regimes_probs,dims=2)
end

module Temp
struct Lattice
    likehood:: Float64
    nodes::  Vector{Vector{Float64}}
    probabilities
end
end

function hmm_lattice(data, n_nodes)

    means, c_cov = cluster_moments(data,n_nodes)

    guess_dist = [MvNormal(means[:,s],c_cov[:,:,s]) for s in 1:n_nodes]
    init_guess = guess_init(n_nodes)
    guess_matrix = guess_tmatrix(n_nodes)
    hmm_guess = HMM(init_guess, guess_matrix, guess_dist);
    hmm_est, likehood = baum_welch(hmm_guess, eachcol(data);max_iterations=200);
    nodes = mean.(obs_distributions(hmm_est))
    return Temp.Lattice(last(likehood), nodes, transition_matrix(hmm_est))
end

