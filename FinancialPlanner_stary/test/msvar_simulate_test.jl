using Revise
using Test

#includet("../_wip/msvar_temp.jl") 


@testset "simulate MSVAR" begin 

  X = [.12, .06]

  transition_matrix = [.9 .1;.2 .8]
  
  states_zero = [.67, .33]
  
  
  Β1 = [.08 .3 .2; .03 .1 .6]
  Σ1 = cor2cov([1 .3; .3 1], [.15, .03])
  Β2 = [.085 .35 .25; .035 .15 .63]
  Σ2 = cor2cov([1 .35; .35 1], [.16, .035])
   
  
  regimes = simulate_next_regimes(states_zero, transition_matrix, 2_000)
  
  result = simulate_returns_step(regimes, [Β1, Β2], [Σ1, Σ2], repeat(X, 1, 2_000))
  
  @test mean(result, dims=2) ≈ [.132, .081] atol=.009

  result1 = simulate_msvar(transition_matrix, [Β1, Β1], [Σ1, Σ2], X, states_zero, 3, 2_000)
  @test mean(result1[:,1,:], dims=2) ≈ [.132, .081] atol=.009

  regimes2 = simulate_next_regimes_for_previous_regimes(regimes, transition_matrix)

  expected_n_regime_one = (0.7 * 0.9 + 0.3 * 0.2) * 2000
  @test count(i -> i == 1, regimes2 ) ≈ 1380 atol=80

  #TODO test for second period
 

end

@testset "predict" begin
 #TODO: move to different test set
 X = [1., .12, .06]
 Β1 = collect([.08 .3 .2; .03 .1 .6])
 @test predict(Β1,X) ≈ [0.128, 0.078] atol=.001

end