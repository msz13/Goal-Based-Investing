function covariance_posterior(data, scale_prior, d_posterior)
    
    res = diff(data, dims=1)
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)    

end




"""
    data: T x m matrix of observations
    priors: Named tuples of priors
    n_samples: number of samples
    burnin: numper of samples to discart
    thin: skip nth sample
"""

function gibs_sampler(data, priors; burnin = 1000, n_samples=1000, thin=1)

    n_time_steps, n_obs = size(data) 
    n_trends = 2
    n_states = n_trends + n_obs
    p = 1 #number of var lags
    k = n_obs * p #number of var variables 

    n_draws = burnin + n_samples



    #posterior degrees pf freedom for trend covariance matrix
    dτ_post = n_time_steps - p + priors.trend_covariance_df

    #posterior degrees pf freedom for cycle covariance matrix
    dc_post = n_time_steps - p + priors.cycle_covariance_df

    #prior of covariance matrix, for lag one
    λ = priors.cycle_coeff_shrinkage_param
    Ω = λ^2 ./ diag(priors.cycle_covariance_mean)
    Ω_inv = inv(Diagonal(Ω))

    
    trend_covariance_scale = priors.trend_covariance_mean * (priors.trend_covariance_df + n_obs + 1)
    cycle_covariance_scale = priors.cycle_covariance_mean * (priors.cycle_covariance_df + n_obs + 1)

    initial_cycle_covariance = priors.cycle_covariance_mean
    
    # Storage for sampled states and variables
    trends_states = zeros(n_draws, n_time_steps, n_trends)
    cycle_states = zeros(n_draws, n_time_steps, n_trends)
    
    trend_covariance = zeros(n_draws, n_trends, n_trends)
    betas = zeros(n_draws, k*k)
    sigmas = zeros(n_draws, n_obs, n_obs)

    #sample initial parameters values from prior distribution
    trend_covariance[1, :, :] = rand(InverseWishart(priors.trend_covariance_df, trend_covariance_scale))
    betas[1, :] = priors.cycle_coeff_mean #rand(MvNormal(vec(priors.cycle_coeff_mean), reshape(vec(diagm(Ω)), k, k)))
    sigmas[1, :, :] = rand(InverseWishart(priors.cycle_covariance_df, cycle_covariance_scale)) 
    

    for s in 2:n_draws
              
        trends_states[s,:,:], cycle_states[s,:,:] = sample_states(
                                       data, 
                                       reshape(betas[s-1, :], n_obs, k), 
                                       trend_covariance[s-1,:,:], 
                                       sigmas[s-1,:,:], 
                                       priors.initial_trend_mean, 
                                       priors.initial_cycle_mean, 
                                       priors.initial_trend_covariance, 
                                       initial_cycle_covariance)

        trend_covariance[s, :, :] = rand(covariance_posterior(trends_states[s,:,:], trend_covariance_scale, dτ_post))

        betas[s,:], sigmas[s, :, :] = sample_var_params(cycle_states[s,:,:], 1, priors.cycle_coeff_mean, Ω_inv, cycle_covariance_scale, dc_post)       

    end

    t_trends_states = trends_states[burnin+1:thin:end, :, :]
    t_cycle_states =  cycle_states[burnin+1:thin:end, :, :]
    t_trend_covariance = Chains(reshape(trend_covariance[burnin+1:thin:end,:,:], n_samples÷thin, n_trends*n_trends, 1), ["Στ[1]", "Στ[2]", "Στ[3]", "Στ[4]"])
    t_betas = Chains(betas[burnin+1:thin:end,:,:], ["β1", "β2" , "β3", "β4"])
    t_sigmas = Chains(reshape(sigmas[burnin+1:thin:end,:,:], n_samples÷thin, 4, 1), ["Σc[1]", "Σc[2]", "Σc[3]", "Σc[4]"])
    
    return t_trends_states, t_cycle_states, t_trend_covariance, t_betas, t_sigmas 

end