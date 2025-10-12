using LinearAlgebra

σ = [2, 2]

C = [1 0.5
     0.5 1]

Σ = Diagonal(σ)' * C * Diagonal(σ)

P = cholesky(Σ).L 

Φ0 = Matrix(I(2))

e = [1, 0]

Φ0 * P * e

Σ[1,1]^-.5 * Φ0 * Σ * e 

function girf(B::Matrix{Float64}, Σ::Matrix{Float64}, h::Int, shock_var::Int)
    """
    Generalized Impulse Response Function (GIRF) for VAR models
    
    Parameters:
    -----------
    B : Matrix{Float64}
        VAR coefficient matrix of size (K, K*p) where K is number of variables
        and p is the lag order. Contains [A₁ A₂ ... Aₚ]
    Σ : Matrix{Float64}
        Residual covariance matrix of size (K, K)
    h : Int
        Number of periods (horizon) for impulse response
    shock_var : Int
        Index of the variable to be shocked (1 to K)
    
    Returns:
    --------
    Matrix{Float64}
        GIRF matrix of size (h+1, K) where each row represents the response
        at time t, and each column represents a different variable
    """
    
    K = size(Σ, 1)  # Number of variables
    p = size(B, 2) ÷ K  # Lag order
    
    # Initialize GIRF matrix
    girf_matrix = zeros(h + 1, K)
    
    # Shock size: one standard deviation shock scaled by covariance
    σⱼ = sqrt(Σ[shock_var, shock_var])
    eⱼ = zeros(K)
    eⱼ[shock_var] = 1.0
    
    # Generalized impulse: Σ * eⱼ / σⱼ
    shock = Σ * eⱼ / σⱼ
    
    # Period 0: immediate impact
    girf_matrix[1, :] = shock
    
    # Compute companion form if p > 1
    if p > 1
        # Companion matrix
        F = zeros(K * p, K * p)
        F[1:K, :] = B
        F[K+1:end, 1:K*(p-1)] = I(K * (p - 1))
        
        # Extended shock vector
        shock_extended = vcat(shock, zeros(K * (p - 1)))
        
        # Iterate through horizons
        state = shock_extended
        for t in 1:h
            state = F * state
            girf_matrix[t + 1, :] = state[1:K]
        end
    else
        # Simple VAR(1) case
        state = shock
        for t in 1:h
            state = B * state
            girf_matrix[t + 1, :] = state
        end
    end
    
    return girf_matrix
end

B = [.6 .2
     .3 .5]

resp = girf(B, Σ, 20, 1)

