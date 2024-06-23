using StatsBase
using PrettyTables

function returns_summarystats(data::TimeArray,t)
    names = colnames(data)
    returns = transpose(values(data))
    n_assets = size(returns)[1]

    stats = [ Dict(
        :mean => mean(returns[i,:]) * t, 
        :std => std(returns[i,:]) * t^0.5,
        :median => median(returns[i,:] * t),
        :skewness => skewness(returns[i,:]),
        :kurtosis => kurtosis(returns[i,:]),
        :autocor => autocor(returns[i,:],[1])[1],
        :p25th => percentile(returns[i,:],25) * t,
        :p75th => percentile(returns[i,:],75) * t,
        :min => minimum(returns[i,:]) * t,
        :max => maximum(returns[i,:]) * t,
        :sr => (mean(returns[i,:]) * t)/(std(returns[i,:]) * t^0.5),
        ) for i in 1:n_assets ]
        

    short_stats = pretty_table(stats, backend = Val(:html), row_labels = names)
    return short_stats
end

function cor_returns(returns:: TimeArray)
    col = colnames(returns)
    corr = cor(values(returns))
    return pretty_table(corr,header=col, backend = Val(:html), row_labels=col)
end


