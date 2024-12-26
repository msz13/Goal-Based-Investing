module LatticeGeneration

using StatsBase, Clustering

export cluster, clusters_probs, generate_lattice, Lattice, successors_probs

function cluster(data, n_clusters)
    dt = StatsBase.fit(ZScoreTransform, data; dims=2, center=true, scale=true)
    standardized = StatsBase.transform(dt,data)

    clusters = kmeans(standardized,n_clusters)

    destandarised = StatsBase.reconstruct(dt, clusters.centers)
    
    return transpose(destandarised), assignments(clusters)
end

function clusters_probs(cluster_assignments:: Vector{Int64}, n_nodes:: Int64)
    counts(cluster_assignments, n_nodes)/length(cluster_assignments)
   
end

"""
    successors_probs(successors_assignments, ancessors_assignments, n_nodes, node)

Calculates next state probabilities for previous stage node.

successors_assignments: clusters of scenarios of t stage
ancessors_assignments: clusters of scenarios of t + 1 stage
n_nodes - number of nodes
node: node for which next stage probabilies are calculates

"""
function successors_probs(ancessors_assignments, successors_assignments, n_nodes, node)
     return clusters_probs(successors_assignments[ancessors_assignments .== node],n_nodes)
end

mutable struct Lattice
    states:: Array{Float64,3}
    probs:: Array{Matrix{Float64},1}
end




function generate_lattice(scenarios:: Array{Float64,3}, n_nodes, n_stages)

   
    n, n_scenarios, T = size(scenarios)
    
    states =zeros(n_stages, n_nodes, n)
   
    probablities = Array{Float64,2}[[1.0]']

    clusters_assignments = zeros(Int64, T, n_scenarios)

    for t in 1:n_stages
        stage_states, clusters_assignments[t,:] = cluster(scenarios[:,:,t],n_nodes)
        states[t,:,:] .= stage_states
    end

    #calculate probabilities for first stage
    
    probs = clusters_probs(clusters_assignments[1,:], n_nodes)
    
    push!(probablities,probs')

    #calculate probabilities for first stage

    for t in 2:T
        stage_probs = zeros(n_nodes, n_nodes)
        for node in 1:n_nodes
            stage_probs[node,:] .= successors_probs(clusters_assignments[t-1,:],clusters_assignments[t,:],n_nodes,node)
        end
        push!(probablities,stage_probs)
    end

    return Lattice(states, probablities)


end

end