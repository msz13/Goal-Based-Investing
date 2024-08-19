module LatticeGeneration

using StatsBase, Clustering

export cluster, clusters_probs, generate_lattice, Lattice

function cluster(data, n_clusters)
    dt = StatsBase.fit(ZScoreTransform, data; dims=2, center=true, scale=true)
    standardized = StatsBase.transform(dt,data)

    clusters = kmeans(standardized,n_clusters)

    destandarised = StatsBase.reconstruct(dt, clusters.centers)
    
    return transpose(destandarised), assignments(clusters)
end

function clusters_probs(cluster_assignments)
    counts(cluster_assignments) ./length(cluster_assignments)
end

mutable struct Lattice
    states:: Array{Float64,3}
end

function generate_lattice(scenarios:: Array{Float64,3}, n_nodes, n_stages)

    s = sum(scenarios[:,:,21:25]; dims=3)
    s = dropdims(s; dims= 3)

    n = 3 # number of assets
    
    states =zeros(n_stages,n_nodes,n)
    probablities = []

    states_one, clusters = cluster(s,n_nodes)
    probs = clusters_probs(clusters)

    states[1,:,:] .= states_one
    append!(probablities,probs)

    return Lattice(states), probablities


end

end