module FinancialBVAR

using Distributions

    using LinearAlgebra

    export NormalWishartBVARmodel, NormalWishartBVAR, sample_posterior!

    mutable struct NormalWishartBVARmodel
        const Y:: AbstractArray
        const X:: AbstractArray
        const C_OLS:: AbstractArray
        const S_OLS:: AbstractArray
        const df:: Int64
        Σ:: Union{AbstractArray,Missing} 
        Β:: Union{AbstractArray,Missing} 
    end

    function NormalWishartBVAR(data)
        p = 1   #lag
        T,n  = size(data)
        df = T -1 - n
        Y = data[p+1:end,:]
        X = hcat(ones(T-1), data[p:end-1,:])
        C = inv(transpose(X) * X) * transpose(X) * Y
        S = transpose((Y - X*C)) * (Y - X*C)
       
        return NormalWishartBVARmodel(Y, X, C, S, df, missing, missing)
    end

    function sample_posterior!(model :: NormalWishartBVARmodel, n_samples, burnin)
        posterior_beta = zeros(n_samples+burnin,20)
        posterior_sigma = zeros(n_samples+burnin, 4, 4)

       
        for n in 1:n_samples+burnin

            posterior_sigma[n,:,:] = rand(InverseWishart(model.df,model.S_OLS))

            Beta_mean = vec(model.C_OLS)
            Beta_var = kron(posterior_sigma[n,:,:],Hermitian(inv(transpose(model.X) * model.X)))
            posterior_beta[n,:,:] = rand(MvNormal(Beta_mean,Hermitian(Beta_var)))
        
        end               
       
        model.Β = posterior_beta[burnin+1:end,:]
        model.Σ = posterior_sigma[burnin+1:end,:,:]
        
    end

end