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
    probs:: Array{Matrix{Float64},1}
end




function generate_lattice(scenarios:: Array{Float64,3}, n_nodes, n_stages)

   
    n = 3 # number of assets
    
    states =zeros(n_stages,n_nodes,n)
    assignments = zeros_
    probablities = Array{Float64,2}[[1.0]']

    for t in 1:n_stages
        stage_states, clusters_assignments = cluster(scenarios[:,:,1],n_nodes)
        states[t,:,:] .= stage_states
    end

    
    #probs = clusters_probs(clusters)

    
    #push!(probablities,probs')

    

    return Lattice(states, probablities)


end

end