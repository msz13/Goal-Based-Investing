

"""
    minnesota_priors_arraydist(Y; λ1=0.2, λ2=0.5, λ3=1.0, p=2, zero_ownlag=fill(false, N))

Return an `arraydist` object for Minnesota priors over VAR(p) coefficients.

# Arguments
- `Y`: T × N data matrix
- `λ1`: overall tightness
- `λ2`: cross-variable tightness
- `λ3`: constant term tightness
- `p`: number of lags
- `zero_ownlag`: Bool vector length N; if `true`, own lag-1 mean = 0.

# Returns
- `dist`: an `arraydist` of Normal distributions in VAR stacking order:
    for eq = 1..N: [lag1 var1, lag1 var2, ..., lagp varN, constant]
"""
function minnesota_priors(Y; λ1=0.2, λ2=0.5, λ3=1.0, p=2, zero_ownlag=fill(false, size(Y,2)))
    T, N = size(Y)
    @assert length(zero_ownlag) == N "zero_ownlag must be length N"

    σ_y = std.(eachcol(Y))
    k_eq = N * p + 1
    total_params = N * k_eq

    priors = Vector{Normal}(undef, total_params)
    idx = 1
    for eq in 1:N
        for lag in 1:p
            for var in 1:N
                if var == eq
                    μ = (lag == 1 && !zero_ownlag[eq]) ? 1.0 : 0.0
                    σ = λ1 / lag
                else
                    μ = 0.0
                    σ = λ1 * λ2 * (σ_y[eq] / σ_y[var]) / lag
                end
                priors[idx] = Normal(μ, σ)
                idx += 1
            end
        end
        μ_const = 0.0
        σ_const = λ1 * λ3 * σ_y[eq]
        priors[idx] = Normal(μ_const, σ_const)
        idx += 1
    end

    return priors
  
end
