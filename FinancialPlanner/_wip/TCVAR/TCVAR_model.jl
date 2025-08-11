#= function TCVAR(
    params:: NamedTuple,
    initial_observation:: Vector{Float64},
    initial_covariance:: Matrix{Float64}
    )

    (στπ, στr, Φ11, Φ12, Φ21, Φ22, σcπ, σcr) = params

    A = [1 0 0 0
        0 1 0 0
        0 0 Φ11 Φ21
        0 0 Φ21 Φ22
        ]

    B = [1. 0 1. 0
         1. 1. 0 1.
         ]

    return LinearGaussianStateSpaceModel(
        initial_observation,
        initial_covariance,
        A,
        zeros(2),
        diagm([στπ, στr, σcπ, σcr]),
        B,
        zeros((2,2)) 

    )
end
 

 =#

 # Build SSM from parameter matrices (accepts full 2x2 Q_trend and Q_cycle)
function TCVAR(phi::Matrix{Float64}, Q_trend::Matrix{Float64}, Q_cycle::Matrix{Float64})
    # State ordering: [trend_infl, trend_rate, cycle_infl, cycle_rate]
    A = zeros(4,4)
    A[1:2, 1:2] .= I(2)            # trend random walk
    A[3:4, 3:4] .= phi             # cycles VAR(1)

    b = zeros(4)
          
    Q = zeros(4,4)
    Q[1:2, 1:2] .= Q_trend
    Q[3:4, 3:4] .= Q_cycle

    H = [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
    R = zeros(2, 2)

    x0 = zeros(4)
    Σ0 = 100.0 * Matrix(I, 4, 4)
     
    return create_homogeneous_linear_gaussian_model(x0, Σ0, A, b, Q, H, c, R)

end
