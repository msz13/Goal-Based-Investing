module GoalInvestingSDDP

using SDDP, HiGHS, Plots, XLSX, Mustache, PrettyTables
#= include("kmeans_moments_lattice.jl")
using .LatticeGeneration =#
export GoalsData3, asset_management_alm, Optimiasation_Result, describe_result


struct GoalsData3
    minimum_limit:: Vector{Float64}
    acceptable_limit:: Vector{Float64}
    desired_limit::Vector{Float64}
    inflows:: Vector{Float64}
    minimum_utility:: Vector{Float32}
    acceptable_utility:: Vector{Float32}
    desired_utility:: Vector{Float32}
    above_desired_utility:: Vector{Float32}
    initial_wealth:: Float64
    provision:: Float32
end

function asset_management_alm(data:: GoalsData3, scenario_lattice)

  
        model = SDDP.PolicyGraph(
            SDDP.MarkovianGraph(scenario_lattice.probs),
            lower_bound = -64,
            optimizer = HiGHS.Optimizer,
        ) do subproblem, index
            (stage, markov_state) = index

            n_assets = 3           
                  
            assets_returns = scenario_lattice.states           
           
            #cpi = vcat([1], cumprod(ones(25) .+ 0.025))
    
            @variable(subproblem, assets[i = 1:n_assets] .>= 0, SDDP.State, initial_value = 0.0)        
                                 
            @variable(subproblem, assets_buy[i = 1:n_assets] .>= 0)
            @variable(subproblem, assets_sell[i = 1:n_assets]  .>= 0)                           
                 
                       
            @variable(subproblem, consumption >=0)
        
            @variable(subproblem, 0 <= minimum <= data.minimum_limit[stage]) 
            @variable(subproblem, 0 <= acceptable <= data.acceptable_limit[stage] - data.minimum_limit[stage]) 
            @variable(subproblem, 0 <= desired <= data.desired_limit[stage]  - data.acceptable_limit[stage]) 
            @variable(subproblem, 0 <= above_desired)

        
            @constraint(subproblem, minimum + acceptable + desired  + above_desired == consumption)
            
                         
            if stage == 1
                @constraint(subproblem, [i = 1:n_assets], assets_buy[i] - assets_sell[i] == assets[i].out)

                @constraint(subproblem, sum(assets_buy) *(1+data.provision) + consumption == data.initial_wealth) 
                                               
                @stageobjective(subproblem, -(data.minimum_utility[stage] * minimum + data.acceptable_utility[stage] * acceptable + data.desired_utility[stage] * desired + data.above_desired_utility[stage] * above_desired))
                
            elseif 1 < stage 
                @constraint(
                    subproblem,
                    [i = 1:n_assets],
                    (assets_returns[stage-1, markov_state, i]) * assets[i].in + assets_buy[i] - assets_sell[i] == assets[i].out)
                        
                                   
                @constraint(subproblem, sum(assets_buy) *(1+data.provision) - sum(assets_sell) *(1+data.provision) + consumption - data.inflows[stage] == 0) 
    
                @stageobjective(subproblem, -(data.minimum_utility[stage] * minimum + data.acceptable_utility[stage] * acceptable + data.desired_utility[stage] * desired + data.above_desired_utility[stage] * above_desired))
                            
            end
        end       
       
       #@test SDDP.calculate_bound(model) ≈ 1.514 atol = 1e-4
        return model
end

struct Optimistion_Result
    lower_bound:: Float64
    confidence_interval_mu:: Float64
    confidence_interval_ci:: Float64
    consumption_dist:: Array{Float64, 2}
end

function describe_result(model)

    simulations = SDDP.simulate(
    # The trained model to simulate.
    model,
    # The number of replications.
    2000,
    # A list of names to record the values of.
    [:assets, :minimum, :acceptable, :desired, :above_desired, :consumption],
    skip_undefined_variables=true
)

    lb =  round(SDDP.calculate_bound(model), digits=4)

    objectives = map(simulations) do simulation
        return sum(stage[:stage_objective] for stage in simulation)
    end

    μ, ci = round.(SDDP.confidence_interval(objectives), digits=4)

    n_scenarios = 2000
    n_stages = 16
    n_assets = 3

    consumption = zeros(n_scenarios,n_stages)
    assets = zeros(n_assets,n_scenarios,n_stages)

    for (i, scenario) in enumerate(simulations)
        consumption[i,:] = [node[:consumption] for node in scenario]
        for a in 1:3
            assets[a,i,:] = [node[:assets][a].out for node in scenario]
        end
    end
    
    return Optimistion_Result(lb, μ, ci, consumption)

end
    
end