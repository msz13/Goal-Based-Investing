"""
Steady state vector autoregression
   Ψ: steady state mean
   Π: coefficients
   Σ: covariance matrix
"""
struct SSVAR
    Ψ
    Π
    Σ
end