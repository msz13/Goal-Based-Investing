module FinancialBVAR

    using Distributions
    using LinearAlgebra
    using .FinancialBVAR
    using PrettyTables
    using TimeSeries
    using StatsBase
    using MCMCChains

    export NormalWishartBVARmodel, NormalWishartBVAR, sample_posterior!, drift, simulate, model_summary, posterior_summary

    mutable struct NormalWishartBVARmodel
        const var_names:: Vector
        const Y:: AbstractArray
        const X:: AbstractArray
        const C_OLS:: AbstractArray
        const S_OLS:: AbstractArray
        const df:: Int64
        Σ:: Union{AbstractArray,Missing} 
        Β:: Union{AbstractArray,Missing} 
    end

    function NormalWishartBVAR(data:: TimeArray)
        p = 1   #lag
        val = values(data)
        T,n  = size(val)
        df = T -1 - n
        Y = val[p+1:end,:]
        X = hcat(ones(T-1), val[p:end-1,:])
        C = inv(transpose(X) * X) * transpose(X) * Y
        S = transpose((Y - X*C)) * (Y - X*C)
       
        return NormalWishartBVARmodel(colnames(data),Y, X, C, S, df, missing, missing)
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

    function posterior_summary(model:: NormalWishartBVARmodel)
        
        n = length(model.var_names)

        full_var_names = string.(["const"; model.var_names])

        for i in 1:n
            display(string(model.var_names[i]) * " coefficients")
            chn = Chains(model.Β[:,i*(n+1)-n:i*(n+1)],full_var_names)
            df = quantile(chn)
            display(df)
        end  
        
        cov_names = [string.(model.var_names[j]) * "_" * string.(model.var_names[i]) for j in 1:n for i in 1:n]
              
       cov_chn = Chains(reshape(model.Σ,(10000,16)), cov_names)

       display("cov matrix")
       display(quantile(cov_chn))

    end


    
    mutable struct VARModel
        const var_names:: Vector
        const Y:: AbstractArray
        const X:: AbstractArray
        const C:: AbstractArray
        const Σ:: AbstractArray
                
    end

    function VARModel(data::TimeArray)
        p = 1   #lag
        val = values(data)
        T,n  = size(val)
        Y = val[p+1:end,:]
        X = hcat(ones(T-1), val[p:end-1,:])
        C = inv(transpose(X) * X) * transpose(X) * Y
        S = transpose((Y - X*C)) * (Y - X*C) / (T- n -1)
       
        return VARModel(colnames(data), Y, X, C, S) 
    end

    function drift(C,X)
        X = vcat(ones(1),X)
        return C * X 
    end

    function model_summary(model)
        std = sqrt.(diag(model.Σ))
        var_summary = round.(hcat(transpose(model.C),std),digits=4)
        pretty_table(var_summary; backend = Val(:html), header=string.([:const; model.var_names; :std]), row_labels=model.var_names, title="Coefficients")

        cor = cov2cor(model.Σ,std)
        pretty_table(round.(cor,digits=2);backend = Val(:html), header=string.(model.var_names), row_labels=model.var_names, title="Residuals correlations")

    end
    
    """
        simulate(mode,n_steps,n_scenarios)

        return
        result[n_variables,n_scenarios,n_steps]
    """
    function simulate(model, n_steps::Int64, n_scenarios=10_000)
        n = size(model.Y)[2]
        result = zeros(n, n_scenarios,n_steps+1)
        C = transpose(model.C)

        result[:,:,1] = repeat(model.Y[end,:],n_scenarios)

        for t in 2:n_steps+1
            for s in 1:n_scenarios
                result[:,s,t] = rand(MvNormal(drift(C,result[:,s,t-1]),model.Σ),)
            end
        end

        return result

    end


end


