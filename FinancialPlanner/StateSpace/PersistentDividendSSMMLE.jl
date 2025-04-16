using Revise
using StateSpaceModels, LinearAlgebra, Distributions


includet("PersistentDividendSSMMLE_model.jl")


y = rand(MvNormal([.05,.05,.05,.05,.05], Matrix(I(5))), 100)

create_system(collect(y'))


params_names = handle_model_names()
println(params_names)



model = PersistentDividendSSM(y)



initial_hyperparameters!(model)

m_new = fill_model_system!(model)

f = default_filter(model)
scenarios = simulate_scenarios2(model, 250, 1)

print_results(model)

constrain_hyperparameters!(model)

unconstraint_hyperparameters!(model)