
  
    mutable struct VARModel
        const var_names:: Vector
        const Y:: AbstractArray
        const X:: AbstractArray
        const C:: AbstractArray
        const Σ:: AbstractArray
                
    end

    function VARModel(data::TimeArray, p=1, intercept=true)

        T, n_assets = size(data)
        Y, X = prepare_var_data(values(data), 1, Matrix{Float64}(undef, 0, 0), true)

        C = inv(transpose(X) * X) * transpose(X) * Y
        S = transpose((Y - X*C)) * (Y - X*C) / (T- n_assets -1)
       
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
        result = zeros(n, n_steps+1, n_scenarios)
        C = transpose(model.C)

        result[:,1,:] = repeat(model.Y[end,:],n_scenarios)

        for t in 2:n_steps+1
            for s in 1:n_scenarios
                result[:,t, s] = rand(MvNormal(drift(C,result[:,t-1, s]),Hermitian(model.Σ)),)
            end
        end

        return result

    end
