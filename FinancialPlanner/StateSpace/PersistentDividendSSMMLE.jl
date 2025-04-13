using Revise
using StateSpaceModels, LinearAlgebra, Distributions


includet("PersistentDividendSSMMLE_model.jl")


y = rand(MvNormal([.05,.05,.05,.05,.05], Matrix(I(5))), 100)

create_system(collect(y'))



params_names = handle_model_names()
println(params_names)



model = PersistentDividendSSM(y)

print_results(model)



m_new = fill_model_system!(model)

