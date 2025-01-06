#using Test
using .ESGModels
using Random
#includet("../_wip/msvar_temp.jl") 



@testset "simulate MSVAR" begin 

  #Random.seed!(13)
  X = [.12, .06]

  transition_matrix = [.85 .15;.2 .8]
  
  states_zero = [.67, .33]
  
  
  Β1 = [.09 .3 .2; .03 .1 .6]
  Σ1 = cor2cov([1 .3; .3 1], [.08, .03])
  Β2 = [-.02 .35 .25; .035 .15 .63]
  Σ2 = cor2cov([1 .35; .35 1], [.18, .035])

  @testset "expected_regimes" begin
    
    t = 3
    expected_regimes = ESGModels.expected_regimes(states_zero, transition_matrix, t)
    
    @test expected_regimes[1,:] ≈ [.6355, .3645] atol=.0001
    @test expected_regimes[2,:] ≈ [.6131, .3869] atol=.0001
    @test expected_regimes[3,:] ≈ [.5985, .4015] atol=.0001

  end


  @testset "simulate_regimes" begin 

    n_steps = 3
    n_scenarios = 300
    expected_regimes = ESGModels.expected_regimes(states_zero, transition_matrix, n_steps)


    regimes = ESGModels.simulate_regimes(states_zero, transition_matrix, n_steps, n_scenarios)

   
    @test size(regimes) == (n_steps, n_scenarios)
    @test count(i -> i == 1, regimes[1,:]) ≈ n_scenarios*expected_regimes[1] atol=25
    @test count(i -> i == 1, regimes[2,:]) ≈ n_scenarios*expected_regimes[2]  atol=25
    @test count(i -> i == 1, regimes[3,:]) ≈ n_scenarios*expected_regimes[3]  atol=25

    #TODO-zrobić jakis sensowny test
    # test czy program robi co powinien, czyli generauje sample
    #czy dystrybucja spełnia, co chcemy
    #czy dystrubucja spełnia nalityczne oczekiwania co do dystrybucji
    #test regresji czy test zawsze zwraca to co chcemy z seed
    #test z mockiem samplera

  end
  
  
  
  @testset "simulate_returns" begin

    n_variables = 2
    n_steps = 2
    n_scenarios = 500
   

    simulated_regimes = ESGModels.simulate_regimes(states_zero, transition_matrix, n_steps, n_scenarios)

    simulated_returns = ESGModels.simulate_msvar_returns(simulated_regimes, [Β1, Β2], [Σ1, Σ2], X, n_steps, n_scenarios)

    @test size(simulated_returns) == (n_variables, n_steps, n_scenarios)
    
    @test mean(simulated_returns[:, 1, :], dims=2) ≈ [.1012, .0827] atol=0.01
    @test mean(simulated_returns[:, 2, :], dims=2) ≈ [.0876, .0937] atol=0.01
    #TODO analical mean for t2
  
  end
#= 
  @test mean(result, dims=2) ≈ [.132, .081] atol=.009

  result1 = ESGModels.simulate_msvar(transition_matrix, [Β1, Β1], [Σ1, Σ2], X, states_zero, 3, 2_000)
  @test mean(result1[:,1,:], dims=2) ≈ [.132, .081] atol=.009

  regimes2 = ESGModels.simulate_next_regimes_for_previous_regimes(regimes, transition_matrix)

  expected_n_regime_one = (0.7 * 0.9 + 0.3 * 0.2) * 2000
  @test count(i -> i == 1, regimes2 ) ≈ 1380 atol=80 =# 

  #TODO test for second period
 

end

@testset "predict" begin
 #TODO: move to different test set
 X = [1., .12, .06]
 Β1 = collect([.08 .3 .2; .03 .1 .6])
 @test ESGModels.predict(Β1,X) ≈ [0.128, 0.078] atol=.001

end