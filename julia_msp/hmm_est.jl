using Clustering
using StatsBase
using Distributions
using TimeSeries
using HiddenMarkovModels


function cluster_moments(data, n_clusters)
    dt = fit(ZScoreTransform, data, dims=2)
    standarized = StatsBase.transform(dt,data)

    clusters = kmeans(standarized,n_clusters)
    c = assignments(clusters)

    cluster_means = zeros(size(data)[1],n_clusters)
    cluster_cov = zeros(size(data)[1],size(data)[1],n_clusters)

    for cluster in 1:n_clusters
        cluster_means[:,cluster] = mean(data[:,c .== cluster],dims=2) 
        cluster_cov[:,:,cluster] = cov(data[:,c .== cluster],dims=2) 
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


function hmm_est(returns, n_nodes)

    data = transpose(values(returns))
    
    means, c_cov = cluster_moments(data,n_nodes)

    guess_dist = [MvNormal(means[:,s],c_cov[:,:,s]) for s in 1:n_nodes]
    init_guess = guess_init(n_nodes)
    guess_matrix = guess_tmatrix(n_nodes)
    hmm_guess = HMM(init_guess, guess_matrix, guess_dist);
    hmm_est, likehood = baum_welch(hmm_guess, eachcol(data);max_iterations=200);
    
    return hmm_est, last(likehood)
end

#= 
function simulate_hmm(hmm,n_steps, n_scenarios)
    simulations = zeros(n_scenarios,n_steps)
    for s in 1:n_scenarios
        simulations[s,:] .= rand(hmm,n_steps)[2]
    end
    return simulations
end =#

function simulate_hmm(hmm, n_assets, n_steps, n_scenarios)
    simulations = zeros(5,n_steps)
   
    random = rand(hmm,n_steps)[2]
    #= for t in 1:n_steps
        for asset in 1:n_assets
            simulations[asset,t] = random[t][asset]
        end
    end =#
    
    return random
end 