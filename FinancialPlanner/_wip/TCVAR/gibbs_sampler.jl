"""
    data: T x m matrix of observations
    priors: Named tuples of priors
    n_samples: number of samples
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
    Ω = [λ^2, λ^2] ./ diag(priors.cycle_covariance_mean)
    Ω_inv = inv(Diagonal(Ω))

    
    trend_covariance_scale = priors.trend_covariance_mean * (priors.trend_covariance_df + n_obs + 1)
    cycle_covariance_scale = priors.cycle_covariance_mean * (priors.cycle_covariance_df + n_obs + 1)

    initial_cycle_covariance = priors.cycle_covariance_mean
    
    # Storage for sampled states and variables
    state_smoothed_samples = zeros(n_draws+1, n_time_steps, n_states) #TODO uporzadkowac, co jest samplowane dla states dla initial
    sampled_trend_covariance = zeros(n_draws+1, n_trends, n_trends)
    betas = zeros(n_draws, k*k)
    sigmas = zeros(n_draws, n_obs, n_obs)

    #sample initial parameters values form prior distribution
    sampled_trend_covariance[1, :, :] = rand(InverseWishart(priors.trend_covariance_df, trend_covariance_scale))
    betas[1, :] = zeros(4) #MvNormal(vec(priors.cycle_coeff_mean), reshape(vec(diagm(Ω)), 4,4))
    sigmas[1, :, :] = Diagonal([.00016, .00016]) # rand(InverseWishart(priors.cycle_covariance_df, priors.cycle_covariance_scale)) 
    

    for s in 2:n_draws
        model = tc_var(
                reshape(betas[s-1, :], n_obs, k), 
                sampled_trend_covariance[s-1,:,:], 
                sigmas[s-1,:,:],       
                priors.initial_trend_mean, 
                priors.initial_cycle_mean,
                priors.initial_trend_covariance,
                initial_cycle_covariance                              
                )
        
        state_smoothed_samples[s, :, :] = carter_kohn_sampler(model, data)

        trends_states = state_smoothed_samples[s, :, [1,2]]
        cycle_states =  state_smoothed_samples[s, :, [3,4]]

        sampled_trend_covariance[s, :, :] = rand(covariance_posterior(trends_states, trend_covariance_scale, dτ_post))

        betas[s,:], sigmas[s, :, :] = sample_var_params(cycle_states, 1, priors.cycle_coeff_mean, Ω_inv, cycle_covariance_scale, dc_post)       

    end

    return state_smoothed_samples[burnin+1:thin:end, :, :], Chains(reshape(sampled_trend_covariance[burnin+1:thin:end,:,:], n_samples÷thin+1, n_trends*n_trends, 1), ["Στ[1]", "Στ[2]", "Στ[3]", "Στ[4]"]), Chains(betas[burnin+1:thin:end,:,:], ["β1", "β2" , "β3", "β4"]),  Chains(reshape(sigmas[burnin+1:thin:end,:,:], n_samples÷thin, 4, 1), ["Σc[1]", "Σc[2]", "Σc[3]", "Σc[4]"])

end