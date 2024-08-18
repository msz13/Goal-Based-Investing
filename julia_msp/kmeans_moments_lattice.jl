module Lattice

using StatsBase, Clustering

export cluster

function cluster(data, n_clusters)
    dt = StatsBase.fit(ZScoreTransform, data; dims=2, center=true, scale=true)
    standardized = StatsBase.transform(dt,data)

    clusters = kmeans(standardized,n_clusters)

    destandarised = StatsBase.reconstruct(dt, clusters.centers)

    probs = counts(clusters)/sum(counts(clusters)) 

    return destandarised, probs
end


end