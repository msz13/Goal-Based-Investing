
function generate_data(T=100, regimes=2, dim=2)
    Random.seed!(42)
    states = rand(1:regimes, T)
    data = zeros(T, dim)
    
    params = [
        ([0.09, 0.03], [0.3 0.2; 0.2 0.6], [0.0064 0.00072; 0.00072 0.0009] .+ 1e-12),
        ([-0.02, 0.045], [0.4 0.25; 0.15 0.43], [0.0324 0.002205; 0.002205 0.001225])
    ]
    
    for t in 2:T
        state = states[t]
        mean, coef, cov = params[state]
        data[t, :] = mean + coef * data[t-1, :] + rand(MvNormal([0, 0], cov))
    end
    
    return data, states
end

@model function switching_model(data, regimes=2)
    T, dim = size(data)
    
    # State sequence.
    s = tzeros(Int, T)

    # Transition matrix.
    P = Vector{Vector}(undef, regimes)

    # Emission matrix.
    mus = Vector(undef, regimes)
    betas = Vector(undef, regimes)
    sigmas = Vector(undef, regimes)
  
    
    # Regime-specific parameters
    for r in 1:regimes
        P[r] ~ Dirichlet(ones(regimes)/regimes)
        mus[r] ~ filldist(Normal(0,1), dim)
        betas[r] ~ filldist(Normal(0,1), dim, dim)
        sigmas[r] ~ InverseWishart(dim + 1, Matrix(I(dim)))
    end

    s[1] ~ Categorical(ones(regimes)/regimes)

    # Observation likelihood
    for t in 2:T
        s[t] ~ Categorical(vec(P[s[t - 1]]))
        mean = mus[s[t]] + betas[s[t]] * data[t-1, :]
        data[t, :] ~ MvNormal(mean, sigmas[s[t]])
    end
end


@model function var_model(data, regimes=2)
    T, dim = size(data)
 

    # Emission matrix.
    mus ~ filldist(Normal(0,1), dim)
    betas ~ filldist(Normal(0,1), dim, dim)
    sigmas ~ InverseWishart(dim + 1, Matrix(I(dim)))
   
    # Observation likelihood
    for t in 2:T
        mean = mus + betas * data[t-1, :]
        data[t, :] ~ MvNormal(mean, sigmas)
    end
end



@model function BayesHmm(y, K)
    # Get observation length.
    N = length(y)

    # State sequence.
    s = tzeros(Int, N)

    # Emission matrix.
    m = Vector(undef, K)

    # Transition matrix.
    T = Vector{Vector}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i in 1:K
        T[i] ~ Dirichlet(ones(K) / K)
        m[i] ~ Normal(i, 0.5)
    end

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i in 2:N
        s[i] ~ Categorical(vec(T[s[i - 1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end
end;