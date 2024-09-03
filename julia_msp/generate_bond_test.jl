using Revise

includet("generate_bond_temp.jl")

using .Bonds
using Test

@testset "One period bond_return" begin
    @test Bonds.duration(0.02,10) ≈ 9.02 atol=0.01
    @test Bonds.convexity(0.02,10) ≈ 90.85 atol=0.01
    @test bond_return(0.01, 0.01, 10, 1) == 0.01
    @test bond_return(0.01, 0.01, 10, 4) == 0.01/4
    @test bond_return(0.02,0.01,10,12) ≈ -0.0849 atol=0.0001
    @test bond_return(0.01,0.02,10,12) ≈ 0.1015 atol=0.0001
end

@testset "Bond returns for scenarios" begin
   scenarios1 = [0.01 0.02 0.025; 0.03 0.025 0.01]
   scenarios2 = [0.01 0.01 0.01; 0.03 0.03 0.03]
   expected1 = [-0.0849 -0.04123; 0.0476 0.1555]
   expected2 = [0.005 0.005; 0.015 0.015]

   @test size(scenarios_bond_returns(scenarios1, 10, 12)) == (2,2)

   @test scenarios_bond_returns(scenarios1, 10, 12) ≈ expected1 atol = 0.001
   @test scenarios_bond_returns(scenarios2,10, 2) ≈ expected2 atol = 0.001

end
   
