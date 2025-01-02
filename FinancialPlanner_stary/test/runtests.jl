#using Revise
using Test

@testset "FinancialPlanner.jl" begin
    include("msvar_fit_test.jl")
    include("msvar_simulate_test.jl")
end
