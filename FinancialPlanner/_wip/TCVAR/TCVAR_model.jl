 
using ForwardDiff

function cholesky_decomposition(x::AbstractMatrix) 
    if isposdef(x)
        return cholesky(x)
    elseif all(i -> i >= 0.0, eigvals(x))
        size_matrix = size(x, 1)
        chol_x = cholesky(x .+ I(size_matrix) .* floatmin())

        chol_x = ForwardDiff.ignore_derivatives() do
            chol_x.L[:, :]  = round.(Float64, chol_x.L; digits = 10)
            chol_x.U[:, :]  = round.(Float64, chol_x.U; digits = 10)
            chol_x.UL[:, :] = round.(Float64, chol_x.UL; digits = 10)
            chol_x
        end
        return chol_x
    else
        @error("Matrix is not positive definite or semidefinite. Cholesky decomposition cannot be performed.", x)
    end
end

# for Kalman filter stability
symmetrize!(A::Symmetric) = A
symmetrize!(A) = Symmetric(A)
symmetrize!(A::Number) = A

 
 # Build SSM from parameter matrices (accepts full 2x2 Q_trend and Q_cycle)
function TCVAR(phi::AbstractMatrix, Q_trend::AbstractMatrix, Q_cycle::AbstractMatrix)
    # Infer numeric type from phi (covers Float64 and Dual)
    T = promote_type(eltype(phi), eltype(Q_trend), eltype(Q_cycle))

    # Transition matrix A
    A = zeros(T, 4, 4)
    A[1:2, 1:2] .= one(T) .* I(2)    # trend random walk
    A[3:4, 3:4] .= phi               # cycle VAR(1)

    # Transition intercept
    b = zeros(T, 4)

    # Process noise covariance Q
    Q = zeros(T, 4, 4)
    Q[1:2, 1:2] .= Q_trend
    Q[3:4, 3:4] .= Q_cycle
    #z[diagind(z)] .+= eps()

    # Observation matrix H
    H = T[ 1  0  1  0
           1  1  0  1
         ]

    # Observation noise covariance R (tiny to avoid singularity)
    #R = Matrix{T}(I, 2, 2) * eps(T)
    R = zeros(T,2,2)

    # Initial state mean and covariance
    x0 = zeros(T, 4)
    Σ0 = Matrix{T}(I, 4, 4) * T(100.0)

    # Observation intercept
    c = zeros(T, 2)

    return create_homogeneous_linear_gaussian_model(x0, Σ0, A, b, PDMat(Q, cholesky_decomposition(symmetrize!(Q))) , H, c, PDMat(R, cholesky_decomposition(symmetrize!(R))))
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