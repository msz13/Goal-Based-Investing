import Pkg

Pkg.activate("FinancialPlanner/StateSpace/StateSpaceModels.jl-master/StateSpaceModels")

using Revise





#includet("persistent_dividend_SSMMLE_model.jl")
#includet("StateSpaceModels.jl-master/src/StateSpaceModels.jl")
using LinearAlgebra, Distributions
using StateSpaceModels

fake_y = rand(MvNormal([.05,.05,.05,.05,.05], Matrix(I(5))), 200)

Θ1 = [.333, .343, .003, .528, .234]
Θ2 = [.338, .286, -.130, -.012, .625]
R = [1.45*10^-1, 1.59*10^-14, 3.01*10^-13, 7.43*10^-10, 3.78*10^-12 ]
#Q = diagm([4.66 * 10^-2, 2.34 * 10^-2, 5.51 * 10^-2, 2.2, 5.0 * 10^-1, 2.53 * 10^-1, 62.11, 2.09])
Q = [4.66 * 10^-2, 2.34 * 10^-2, 5.51 * 10^-2, 2.2, 5.0 * 10^-1, 2.53 * 10^-1, 62.11, 2.09] ./ 100

test_model = create_system(collect(fake_y'), (Θ1 = Θ1, Θ2 = Θ2, R=R, Q=Q, n=40, ρ= 0.967))

test_data = simulate(test_model, zeros(13), 200)


model = PersistentDividendSSM(collect(test_data'))

filter = default_filter(model)


size(model.system.y)


tuple_with_model_type = Tuple{typeof(model)}

m1 = hasmethod(StateSpaceModels.default_filter, tuple_with_model_type)
m2 = hasmethod(StateSpaceModels.initial_hyperparameters!, tuple_with_model_type)
m3 = hasmethod(StateSpaceModels.constrain_hyperparameters!, tuple_with_model_type)
m4 = hasmethod(StateSpaceModels.unconstrain_hyperparameters!, tuple_with_model_type)
m5 = hasmethod(StateSpaceModels.fill_model_system!, tuple_with_model_type)

supertype(typeof(model))

methods(has_fit_methods)
methods(default_filter)

fit!(model)

StateSpaceModels.has_fit_methods(typeof(model))

