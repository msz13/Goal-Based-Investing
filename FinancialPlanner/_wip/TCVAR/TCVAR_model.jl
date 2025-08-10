function TCVAR(
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

    B = [1 0 1 0
         1 1 0 1
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
 

