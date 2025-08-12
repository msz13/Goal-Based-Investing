 # Build SSM from parameter matrices (accepts full 2x2 Q_trend and Q_cycle)
function TCVAR(phi::Matrix, Q_trend::Matrix, Q_cycle::Matrix)
    # State ordering: [trend_infl, trend_rate, cycle_infl, cycle_rate]
    A = zeros(4,4)
    A[1:2, 1:2] .= I(2)            # trend random walk
    A[3:4, 3:4] .= phi             # cycles VAR(1)

    b = zeros(4)
          
    Q = zeros(4,4)
    Q[1:2, 1:2] .= Q_trend
    Q[3:4, 3:4] .= Q_cycle

    H = [1. 0. 1. 0
         1. 1. 0. 1.
        ]

    R = zeros(2,2) + I(2) * eps()
    

    #TODO sample initial state from priors. For trend pre sample mean, variance?, dor cycle zero mean, unconditional var variance
    x0 = zeros(4)
    Σ0 = 100.0 * Matrix(I, 4, 4)

    c = zeros(2)
     
    return create_homogeneous_linear_gaussian_model(x0, Σ0, A, b, Q, H, c, R)

end


@model function trendcycle_model(Y::AbstractVector; k_tr::Real=4.0, Ψ_tr::AbstractMatrix=Matrix(I,2,2), k_cyc::Real=4.0, Ψ_cyc::AbstractMatrix=Matrix(I,2,2))
    # Inverse-Wishart priors (Distributions.jl supports InverseWishart)
    Q_tr ~ InverseWishart(k_tr, Ψ_tr)
    Q_cyc ~ InverseWishart(k_cyc, Ψ_cyc)

    # Observation noise std devs (positive)
    σ_obs1 ~ Normal(0.0, 1.0)
    σ_obs2 ~ Normal(0.0, 1.0)



    # Cycle VAR(1) coefficients
    φ11 ~ Normal(0.0, 1)
    φ12 ~ Normal(0.0, 1)
    φ21 ~ Normal(0.0, 1)
    φ22 ~ Normal(0.0, 1)

 
    phi = [φ11 φ12; φ21 φ22]
  
    tcvar_model = TCVAR(phi, Q_tr, Q_cyc)

    # compute marginal log-likelihood via Kalman filter
    _, ll = GeneralisedFilters.filter(tcvar_model, KF(), Y)
    Turing.@addlogprob!(ll)
end