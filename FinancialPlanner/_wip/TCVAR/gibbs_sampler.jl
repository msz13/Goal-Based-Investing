
function gibs_sampler(data, priors, n_samples)

    n_time_steps = size(data, 1)
    n_states = 4

    true_var_coeff = [.3 .1;  .4 .6]
    true_trend_covariance =  diagm([.02/10, .01/10] .^2) 

    p = -0.7 * .015 * 0.011

    true_cycle_covariance = [(.015)^2 p
                             p (.011)^2]

     
    true_initial_trend_mean = [.02, .01]
    true_initial_trend_covariance = Matrix(I, 2,2)
    true_initial_cycle_mean = [.0, .0]
    true_initial_cycle_covariance = [(.015)^2 0
                                     0 (.011)^2]   
    
    # Storage for sampled states
    state_smoothed_samples = zeros(n_samples, n_time_steps, n_states)
    

    for s in 1:n_samples
        model = tc_var(true_var_coeff, 
                true_trend_covariance, 
                true_cycle_covariance, 
                priors.initial_trend_mean, 
                true_initial_cycle_mean, 
                true_initial_trend_covariance, 
                true_initial_cycle_covariance
                )
        state_smoothed_samples[s, :, :] = carter_kohn_sampler(model, data)
    end

    return state_smoothed_samples 

end