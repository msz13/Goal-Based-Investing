using StateSpaceModels


#TODO handle names
#TODO initiate model
#TODO initial hiperparameters
#TODO constraint hyperparamenters
#TODO unconstraint hyperparamenters
#TODO defoult filter
#TODO reinstantiate
#TODO has egzegonous
#TODO fill model system


mutable struct PersistentDividendSSM <: StateSpaceModel
    hyperparameters:: StateSpaceModels.HyperParameters{Float64}
    system::LinearMultivariateTimeInvariant    
    results:: StateSpaceModels.Results

    function PersistentDividendSSM(y::Matrix{Fl}) where Fl
              
        system = create_system(y)
        names = handle_model_names()
        hyperparameters = StateSpaceModels.HyperParameters{Fl}(names)

        return new(hyperparameters, system, StateSpaceModels.Results{Fl}()) 
    end
end


 function handle_model_names()
    
    state_names = ["dp", "rp", "πp", "da", "ra", "πa", "ea", "τa"]
    observations_names = ["p", "il", "is", "d", "π" ]

    ar_names = ["θ$(lag)_$(v)" for lag in 1:2 for v in state_names[:4:end]]
    measurement_sigmas = ["σ_$(n)" for n in observations_names]
    states_sigmas = ["v$(n)" for n in state_names]
    
    return [ar_names; measurement_sigmas; states_sigmas]
end



 function create_system(y::Matrix)
    
    Fl = Float64
    T = buildF(ones(5), ones(5))
    n = 40
    ρ = 0.967
    Z = buildH(n,ρ, ones(5), ones(5))

    d = zeros(Fl, 5)
    c = zeros(Fl, 13)
    H = diagm(ones(5))
    Q = zeros(13,13)
    Q[1:8,1:8] = diagm(ones(8))
    R = ones(13,13)
    system = LinearMultivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)
    return system
 end


function buildF(Θ1, Θ2)
    return [I(3) zeros(3, 5) zeros(3, 5)
        zeros(5, 3) diagm(Θ1) diagm(Θ2)
        zeros(5, 3) I(5) zeros(5, 5)
    ]
end


function buildH(n, ρ, Θ1, Θ2)
    Hp = [1 / (1 - ρ), -1 / (1 - ρ), 0, Θ1[1] / (1 - ρ * Θ1[1]), -Θ1[2] / (1 - ρ * Θ1[2]), 0, -Θ1[4] / (1 - ρ * Θ1[4]), 0, Θ2[1] / (1 - ρ * Θ2[1]), -Θ2[2] / (1 - ρ * Θ2[2]), 0, -Θ2[4] / (1 - ρ * Θ2[4]), 0]'
    Hin = [0, 1, 1, 0, (1 / n) * ((1 - Θ1[2]^n) / (1 - Θ1[2])) * Θ1[2], (1 / n) * ((1 - Θ1[3]^n) / (1 .- Θ1[3])) * Θ1[3], 0, (1 / n) * ((1 - Θ1[5]^(n - 1)) / (1 .- Θ1[5])) * Θ1[5], 0, (1 / n) * ((1 - Θ2[2]^n) / (1 - Θ2[2])) * Θ2[2], (1 / n) * ((1 - Θ2[3]^n) / (1 .- Θ2[3])) * Θ2[3], 0, (1 / n) * ((1 - Θ2[5]^(n - 1)) / (1 .- Θ2[5])) * Θ2[5]]'
    Hi = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]'
    Hd = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]'
    Hπ = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]'

    H = [Hp
        Hin
        Hi
        Hd
        Hπ
    ]

    return H
end


function buildQ(Q)
    result= zeros(13, 13)
    cov_matrix = diagm(Q)
    result[1:8, 1:8] = cov_matrix
    return result
end



function default_filter(model::PersistentDividendSSM)
    Fl = StateSpaceModels.typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return MultivariateKalmanFilter(
        size(model.system.y, 2), a1, P1, num_states(model), steadystate_tol
    )
end



function initial_hyperparameters!(model:: PersistentDividendSSM)
    names = get_names(model)
    Fl = StateSpaceModels.typeof_model_elements(model)
    initial_hyperparameters = Dict{String,Fl}()
    
    for Θ1 in get_Θ1(names)
        initial_hyperparameters[Θ1] = 1.
    end

    for Θ2 in get_Θ2(names)
        initial_hyperparameters[Θ2] = 1.
    end

    for σ in get_sigmas(names)
        initial_hyperparameters[σ] = 1.
    end

    for v in get_state_variances(names)
        initial_hyperparameters[v] = 1.
    end

    set_initial_hyperparameters!(model, initial_hyperparameters)

    return nothing

end

get_Θ1(names) = filter(x -> occursin("θ1", x), names)
get_Θ2(names) = filter(x -> occursin("θ1", x), names)
get_sigmas(names) = filter(x -> occursin("σ", x), names)
get_state_variances(names) = filter(x -> occursin("v", x), names)


function constrain_hyperparameters!(model::PersistentDividendSSM)
    names = get_names(model)
     
    for Θ1 in get_Θ1(names)
        constrain_box!(model, Θ1, -.9999, .9999)
    end

    for Θ2 in get_Θ2(names)
        constrain_box!(model, Θ2, -.9999, .9999)
    end

    for σ in get_sigmas(names)
        constrain_variance!(model, σ)
    end

    for v in get_state_variances(names)
        constrain_variance!(model, v)
    end

    return model
end

function unconstraint_hyperparameters!(model::PersistentDividendSSM)
    names = get_names(model)
     
    for Θ1 in get_Θ1(names)
        constrain_box!(model, Θ1, -.9999, .9999)
    end

    for Θ2 in get_Θ2(names)
        constrain_box!(model, Θ2, -.9999, .9999)
    end

    for σ in get_sigmas(names)
        constrain_variance!(model, σ)
    end

    for v in get_state_variances(names)
        constrain_variance!(model, v)
    end

    return model
end


function fill_model_system!(model::PersistentDividendSSM)
    
    names = get_names(model)
      
    Θ1_vars = [get_constrained_value(model, n) for n in get_Θ1(names)]
    Θ2_vars = [get_constrained_value(model, n) for n in get_Θ2(names)]
    model.system.T = buildF(Θ1_vars, Θ2_vars)
    n = 40
    ρ = 0.967
    model.system.Z = buildH(n, ρ, Θ1_vars, Θ2_vars)

    model.system.H = diagm([get_constrained_value(model, n) for n in get_sigmas(names)])

    model.system.Q[1:8,1:8] = diagm([get_constrained_value(model, n) for n in get_state_variances(names)])

    return model
end


"""
    simulate_scenarios(
        model::StateSpaceModel, steps_ahead::Int, n_scenarios::Int;
        filter::KalmanFilter=default_filter(model)
    ) -> Array{<:AbstractFloat, 3}

Samples `n_scenarios` future scenarios via Monte Carlo simulation for `steps_ahead`
using the desired `filter`.
"""
function simulate_scenarios2(
    model::PersistentDividendSSM,
    steps_ahead::Int,
    n_scenarios::Int;
    filter::StateSpaceModels.KalmanFilter=default_filter(model),
)
    # Query the type of model elements
    Fl = StateSpaceModels.typeof_model_elements(model)
    fo = kalman_filter(model; filter = filter)
    last_state = fo.a[end]
    num_series = size(model.system.y, 2)

    scenarios = Array{Fl,3}(undef, steps_ahead, num_series, n_scenarios)
    for s in 1:n_scenarios
        scenarios[:, :, s] = simulate(model.system, last_state, steps_ahead)
    end
    return scenarios
end


