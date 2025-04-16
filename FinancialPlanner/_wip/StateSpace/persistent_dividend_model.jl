using StateSpaceInference

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




function PersistentDividendSSM(params::NamedTuple{(:Θ1, :Θ2, :R, :Q)})
    F = buildF(params.Θ1, params.Θ2)
    n = 40
    ρ = 0.967
    H = buildH(n, ρ, params.Θ1, params.Θ2)

    Rmatrix = diagm(params.R)
    Q = buildQ(params.Q)

    model = LinearGaussianStateSpaceModel(zeros(13), zeros(13, 13), F, zeros(13), Q, H, Rmatrix)
    
    return model
end

function persistent_dividend_ssm(params_vector:: AbstractVector)

    Θ1 = params_vector[1:5]
    Θ2 = params_vector[6:10]
    R = params_vector[11:15]
    Q = params_vector[16:23]

    params = (Θ1 = Θ1, Θ2 =Θ1, R = R, Q = Q)
    return PersistentDividendSSM(params)    

end
