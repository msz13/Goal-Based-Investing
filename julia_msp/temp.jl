import Pkg
Pkg.add("ScenTrees")
Pkg.add("FinanceModels")
Pkg.add("EconomicScenarioGenerators")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("CSV")
Pkg.add("DataFrames")

using  CSV, DataFrames, LinearAlgebra

pwd()

inflation_source = CSV.read("sb_gbi/inflation_scenarios.csv",DataFrame)
equity_source = CSV.read("sb_gbi/equity_scenarios.csv",DataFrame)
 
inflation = Matrix(inflation_source[1:5,1:6])
equity = Matrix(equity_source[1:5,1:6])

returns = cat(inflation,equity,dims=3)

returns = permutedims(returns,(2,1,3))

timestamps = [2,2,2]

function sampler()
    t = timestamps
    r = cumprod(returns,dims=3)
    return r
end

sampler()
