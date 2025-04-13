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
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return MultivariateKalmanFilter(
        size(model.system.y, 2), a1, P1, num_states(model), steadystate_tol
    )
end


function fill_model_system!(model::PersistentDividendSSM)
    
    names = get_names(model)

    Θ1names = filter(x -> occursin("θ1", x), names)
    Θ2names = filter(x -> occursin("θ2", x), names)
      
    Θ1_vars = [get_constrained_value(model, n) for n in Θ1names]
    Θ2_vars = [get_constrained_value(model, n) for n in Θ2names]
    model.system.T = buildF(Θ1_vars, Θ2_vars)
    n = 40
    ρ = 0.967
    model.system.Z = buildH(n, ρ, Θ1_vars, Θ2_vars)

    sigmas_names = filter(x -> occursin("σ", x), names)
    model.system.H = diagm([get_constrained_value(model, n) for n in sigmas_names])

    variances_names = filter(x -> occursin("v", x), names)
    model.system.Q[1:8,1:8] = diagm([get_constrained_value(model, n) for n in variances_names])

    return model
end





