using StatsBase
using PrettyTables
using TimeArray


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
        :sr => round((mean(returns[i,:]) * t)/(std(returns[i,:]) * t^0.5), digits=n_digits),
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


function print_percentiles(X, perc, title="")
    years = size(X)[2]
    simulation_perc = zeros(length(perc),years)

    for t in 1:years
        simulation_perc[:,t] = quantile(X[:,t],perc)
    end
    pretty_table(raund.(simulation_perc, digits=4), backend = Val(:html),header=1:years, row_labels=perc, title=title)
end

