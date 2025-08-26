"""
    data: T x m matrix of observations
    priors: Named tuples of priors
    n_samples: number of samples
"""

function gibs_sampler(data, priors, n_samples)

    n_time_steps, n_obs = size(data) 
    n_trends = 2
    n_states = n_trends + n_obs

    #posterior degrees pf freedom for trend covariance matrix
    d_post = n_time_steps - 1 + priors.trend_covariance_df

    true_var_coeff = [.3 .1;  .4 .6]
    true_trend_covariance =  diagm([.02/10, .01/10] .^2) 

    p = -0.7 * .015 * 0.011

    true_cycle_covariance = [(.015)^2 p
                             p (.011)^2]

     
    
    true_initial_trend_covariance = diagm([.01, .01]) #Matrix(I, 2,2)
    true_initial_cycle_mean = [.0, .0]
    true_initial_cycle_covariance = [(.015)^2 0
                                     0 (.011)^2]   
    
    # Storage for sampled states and variables
    state_smoothed_samples = zeros(n_samples+1, n_time_steps, n_states) #TODO uporzadkowac, co jest samplowane dla states dla initial
    sampled_trend_covariance = zeros(n_samples+1, n_trends, n_trends)

    sampled_trend_covariance[1, :, :] = rand(InverseWishart(priors.trend_covariance_df, priors.trend_covariance_mean))
    

    for s in 2:n_samples
        model = tc_var(true_var_coeff, 
                sampled_trend_covariance[s-1,:,:], 
                true_cycle_covariance, 
                priors.initial_trend_mean, 
                true_initial_cycle_mean, 
                true_initial_trend_covariance, 
                true_initial_cycle_covariance
                )
        
        state_smoothed_samples[s, :, :] = carter_kohn_sampler(model, data)

        sampled_trend_covariance[s, :, :] = rand(covariance_posterior(state_smoothed_samples[s, :, [1,2]], priors.trend_covariance_mean, d_post))

    end

    return state_smoothed_samples, Chains(reshape(sampled_trend_covariance, n_samples+1, n_trends*n_trends, 1), ["Στ[1]", "Στ[2]", "Στ[3]", "Στ[4]"])

end