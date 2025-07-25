using StatsBase
using PrettyTables
using TimeSeries


function prepare_var_data(Y::Matrix{Float64}, p::Int, X::Union{Matrix{Float64},Vector{Float64}} = Matrix{Float64}(undef, 0, 0), add_intercept::Bool = false)
    T, n = size(Y)
    Y_lagged = zeros(T - p, n * p)
    for t in (p + 1):T
        Y_lagged[t - p, :] = Y[t-p:t-1, :]'
    end

    predictors = Y_lagged

    if !isempty(X)
        if size(X, 1) != T
            error("The number of rows in X must be equal to the number of rows in Y.")
        end
        X_subset = X[p+1:end, :]
        predictors = hcat(predictors, X_subset)
    end

    if add_intercept
        intercept = ones(T - p, 1)
        predictors = hcat(intercept, predictors)
    end

    return Y[p+1:end, :], predictors
end

function max_drawdown_and_length(returns::Matrix{Float64})
    # returns: Matrix where each column is a scenario, each row a time step
    # Returns a tuple of vectors: (max_drawdowns, max_dd_lengths)
    
    n_scenarios = size(returns, 2)
    max_drawdowns = zeros(Float64, n_scenarios)
    max_dd_lengths = zeros(Int, n_scenarios)
    
    for scenario in 1:n_scenarios
        # Calculate cumulative returns and convert to price path
        cum_returns = cumsum(returns[:, scenario])
        prices = exp.(cum_returns)  # Assuming log-returns, starting price = 1
        
        max_price = prices[1]  # Track historical peak
        max_drawdown = 0.0     # Maximum drawdown (percentage)
        current_dd_length = 0  # Current drawdown length
        max_dd_length = 0      # Maximum drawdown length
        start_idx = 1          # Start index of current drawdown
        
        for t in 2:length(prices)
            if prices[t] >= max_price
                # New peak, reset drawdown
                max_price = prices[t]
                current_dd_length = 0
                start_idx = t
            else
                # In drawdown, update length and drawdown
                current_dd_length = t - start_idx + 1
                current_drawdown = (max_price - prices[t]) / max_price
                max_dd_length = max(max_dd_length, current_dd_length)
                max_drawdown = max(max_drawdown, current_drawdown)
            end
        end
        
        max_drawdowns[scenario] = max_drawdown
        max_dd_lengths[scenario] = max_dd_length
    end
    
    return (max_drawdowns, max_dd_lengths)
end

function returns_summarystats(data::TimeArray,t)
    names = colnames(data)
    returns = transpose(values(data))
    n_assets = size(returns)[1]
    n_digits = 4

    
    stats = [ Dict(
        :mean => round(mean(returns[i,:]) * t, digits=n_digits), 
        :std => round(std(returns[i,:]) * t^0.5, digits=n_digits),
        :median => round(median(returns[i,:]) * t, digits=n_digits),
        :skewness => round(skewness(returns[i,:]), digits=n_digits),
        :kurtosis => round(kurtosis(returns[i,:]), digits=n_digits),
        :autocor => round(autocor(returns[i,:],[1])[1], digits=n_digits),
        :p25th => round(percentile(returns[i,:],25) * t, digits=n_digits),
        :p75th => round(percentile(returns[i,:],75) * t, digits=n_digits),
        :min => round(minimum(returns[i,:]) * t, digits=n_digits),
        :max => round(maximum(returns[i,:]) * t, digits=n_digits),
        :sr => round((mean(returns[i,:]) * t)/(std(returns[i,:]) * t^0.5), digits=n_digits)        
        ) for i in 1:n_assets ]
        

    short_stats = pretty_table(stats, backend = Val(:html), row_labels = names)
    return short_stats
end

function cor_returns(returns:: TimeArray)
    col = colnames(returns)
    corr = cor(values(returns))
    return pretty_table(corr,header=col, backend = Val(:html), row_labels=col)
end

function annualise(scenarios:: Matrix, shift=2)
   
    periods = floor.(Int, size(scenarios)[2]/shift)
    result = zeros(size(scenarios)[1],periods)

    for p in 1:periods
        start = (p-1)*shift+1
        en = p*shift
        result[:,p] .= sum(scenarios[:,start:en],dims=2)
    end 
    return result
   
end


function print_percentiles(X, perc, freq=1, title="")
    scenarios = annualise(X, freq)
    years = size(scenarios)[2]
    simulation_perc = zeros(length(perc),years)

    for t in 1:years
        simulation_perc[:,t] = quantile(scenarios[:,t],perc)
    end
    pretty_table(round.(simulation_perc, digits=4), backend = Val(:html),header=1:years, row_labels=perc, title=title)
end


function sum_returns_between_periods(scenarios::Matrix{Float64}, periods::Vector{Int})
    # Validate inputs
    n_periods, n_scenarios = size(scenarios)
    length(periods) < 2 && error("At least two periods are required")
    all(1 .<= periods .<= n_periods) || error("Invalid period indices")
    issorted(periods) || error("Periods must be sorted in ascending order")
    
    # Initialize output matrix: rows = number of period intervals, cols = number of assets
    n_intervals = length(periods) - 1
    result = zeros(Float64, n_intervals, n_scenarios)
    
    #Sum returns for each interval and asset
    for i in 1:n_intervals
        start_idx = periods[i]
        end_idx = periods[i+1]
        result[i, :] = sum(scenarios[start_idx:end_idx, :], dims=1)
    end
    
    return result
end


function cum_returns_in_periods(scenarios, periods, freq, annualise=false)
    
    n_assets, n_steps, n_scenarios = size(scenarios)
    n_periods = length(periods)

    result = zeros(Float64, n_assets, n_periods, n_scenarios)

    for a in 1:n_assets
        cum_ret  = cumsum(scenarios[a, :, :], dims=1)
        result[a, :, :] =  cum_ret[freq * periods, :] 
        if annualise
            result[a, :, :] = result[a, :, :] ./ periods
        end 
    end

    return result

end

function print_scenarios_summary(scenarios:: Array{Float64, 3}, assets_names, periods)
    n_assets, n_periods, _ =  size(scenarios)
   
    means = zeros(n_periods, n_assets)
    stds = zeros(n_periods, n_assets)
    skew = zeros(n_periods, n_assets)
    kurt = zeros(n_periods, n_assets)

    for a in 1:n_assets
    
        means[:,a] = mean(scenarios[a,:,:], dims=2)
        stds[:,a] = std(scenarios[a, :,:], dims=2)
        
        for t in 1:n_periods
        skew[t,a] = skewness(scenarios[a, t, :])
        end
        
        for t in 1:n_periods
        kurt[t,a] = kurtosis(scenarios[a, t,:])
        end 

    end

    pretty_table(round.(means, digits=4), backend = Val(:html), header=assets_names, row_labels = periods, title="Means")
    pretty_table(round.(stds, digits=4), backend = Val(:html), header=assets_names, row_labels = periods, title="Standard devations")
    pretty_table(round.(skew, digits=4), backend = Val(:html), header=assets_names, row_labels = periods, title="Skewness")
    pretty_table(round.(kurt, digits=4), backend = Val(:html), header=assets_names, row_labels = periods, title="Kurtosis") 

end


function print_scenarios_percentiles(scenarios, perc, periods_names, title="")
    years = size(scenarios, 1)
    simulation_perc = zeros(years, length(perc))

    for t in 1:years
        simulation_perc[t,:] = quantile(scenarios[t,:],perc)
    end
    pretty_table(round.(simulation_perc, digits=4), backend = Val(:html),header=perc, row_labels=periods_names, title=title)
end