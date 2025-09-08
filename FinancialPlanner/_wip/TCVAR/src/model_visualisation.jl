
"""
Compute posterior mean and credible intervals from Carter-Kohn samples
"""
function compute_posterior_statistics(state_smoothed_samples::Array{Float64,3}; credible_level::Float64=0.95)
    n_samples, n_time_steps, n_states = size(state_smoothed_samples)
    
    # Posterior means
    state_smoothed_mean = mean(state_smoothed_samples, dims=1)[1, :, :]
    
    # Credible intervals
    alpha = (1 - credible_level) / 2
    lower_quantile = alpha
    upper_quantile = 1 - alpha
    
    state_smoothed_lower = zeros(n_time_steps, n_states)
    state_smoothed_upper = zeros(n_time_steps, n_states)
    
    for t in 1:n_time_steps
        for s in 1:n_states
            state_samples_ts = state_smoothed_samples[:, t, s]
            state_smoothed_lower[t, s] = quantile(state_samples_ts, lower_quantile)
            state_smoothed_upper[t, s] = quantile(state_samples_ts, upper_quantile)
        end
    end
    
    return state_smoothed_mean, state_smoothed_lower, state_smoothed_upper
end


function plot_variable_states(observations, states, titles)
    
    data = hcat(observations, states)

    plot(data; layout=(3,1), size=(800, 600), title=titles)

end
