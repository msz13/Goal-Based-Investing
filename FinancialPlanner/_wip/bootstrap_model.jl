
#= function bootstrap_simulation(data, block_length)

    T, n = size(data)
    n_steps = block_length

    result = zeros(n, 1, n_steps)
    start_index = 1
    end_index = start_index + block_length - 1

    result[:, 1, :] = data[start_index:end_index, :]'

    
    return result

end =#

using Random

"""
    block_bootstrap_mv(data::AbstractMatrix{T}, B::Int; block_size::Int=5, random_block::Bool=false) where {T}

Performs block bootstrap simulation on multivariate time series data.

# Arguments
- `data`: Matrix of size (T × N), where T is the number of time steps, and N is the number of variables.
- `B`: Number of bootstrap replications.
- `block_size`: Size of each block (default = 5).
- `random_block`: Whether to use random block lengths (default = false).

# Returns
- A 3D array of size (T, N, B), where each slice `[:,:,b]` is one bootstrap sample.
"""
function block_bootstrap(data::Matrix{Tp}, B::Int; block_size::Int=5, random_block::Bool=false) where {Tp}
    T, N = size(data)
    result = Array{Tp, 3}(undef, N, T, B)

    for b in 1:B
        sample_rows = Int[]
        while length(sample_rows) < T
            start_idx = rand(1:(T - block_size + 1))
            this_block_size = random_block ? rand(Geometric(1/block_size)) : block_size
            block_range = start_idx:min(start_idx + this_block_size - 1, T)
            append!(sample_rows, block_range)
        end
        sample_rows = sample_rows[1:T]  # Trim to exact length
        result[:, :, b] = data[sample_rows, :]'
    end

    return result
end


function overlapping_windows(returns, n_steps)
    n_scenarios = size(returns, 1) - n_steps + 1 
    n_assets = size(returns, 2)

    start  = 1

    result = zeros(n_assets, n_steps, n_scenarios)

    for scenario in 1:n_scenarios
        end_idx = start + n_steps - 1
        result[:, :, scenario] = returns[start:end_idx, :]'
        start += 1
    end 

    return result


end