using Clustering
using StatsBase
using Distributions
using TimeSeries
using HiddenMarkovModels
using PrettyTables


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


function simulate_hmm(hmm, n_assets, n_steps, n_scenarios)
    
   
    random = [rand(hmm,n_steps)[2] for s in 1:n_scenarios]
    result = zeros((n_assets,n_scenarios,n_steps))

    for a in 1:n_assets
        for s in 1:n_scenarios
            for step in 1:n_steps
                result[a,s,step] = random[s][step][a]
            end
        end
    end
    
    return  result
end 

function regime_summary(hmm,assets_names, freq)
    
    n_assets = length(assets_names)
    dist = obs_distributions(hmm)
    n_regimes = length(dist)

    #Print regime means
    means = zeros(n_regimes,n_assets)

    for r in 1:n_regimes
        for a in 1:n_assets
            means[r,a] = mean(dist[r])[a]*freq
        end
    end
    
    println("Means")
    pretty_table(means, 
        backend = Val(:html), 
        header=assets_names, 
        show_row_number=true,
        row_number_column_title="Regime", 
        formatters = ft_printf("%5.3f"))

    #Print regime standard deviations
    std = zeros(n_regimes,n_assets)

    for r in 1:n_regimes
        for a in 1:n_assets
            std[r,a] = sqrt(var(dist[r])[a]*freq)
        end
    end
    
    println("Standard deviations")
    pretty_table(std, 
        backend = Val(:html), 
        header=assets_names, 
        show_row_number=true,
        row_number_column_title="Regime", 
        formatters = ft_printf("%5.3f"))

    #Print correlation guess_matrix
    
    for r in 1:n_regimes
        display("Correlations in regime $r")
        pretty_table(cor(dist[r]), 
            backend = Val(:html), 
            header=assets_names, 
            row_labels=assets_names, 
            formatters = ft_printf("%5.3f"))
    end

    println("Regimes transition matrix")
    pretty_table(transition_matrix(hmm), backend = Val(:html), header = 1:n_regimes, show_row_number=true,row_number_column_title="Regime", formatters = ft_printf("%5.3f"))

end