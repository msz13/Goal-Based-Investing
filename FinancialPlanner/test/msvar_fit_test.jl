using Test #, StatsBase
using Revise
includet("../src/ESGModels/ESGModels.jl")
#includet("../src/ESGModels/msvar.jl")
using .ESGModels
using StatsBase



@testset "Hamilton_filter" begin

    Y = [.16 .05; .13 .045]

    X = [1. .12 .06; 1. .16 .05]

    transition_matrix = [.9 .1;.2 .8]

    states_zero = [.67, .33]

    
    Β1 = [.08 .3 .2; .03 .1 .6]'
    Σ1 = cor2cov([1 .3; .3 1], [.15, .03])
    Β2 = [.085 .35 .25; .035 .15 .63]'
    Σ2 = cor2cov([1 .35; .35 1], [.16, .035])


    @testset "calculating next regime" begin
        result = ESGModels.next_regime(states_zero, transition_matrix)
        @test result ≈ [0.669, 0.331] atol=0.001        
    end
    
   
    @testset "likehood for t" begin
        result = ESGModels.likehood_t(Y[1,:], X[1,:], [Β1, Β2], [Σ1, Σ2])

        @test size(result) == (2,)
        @test result[1] ≈ 20.98 atol=0.01 
        @test result[2] ≈ 13.18 atol=0.01         
    end

    @testset "hamilton step" begin

        result = ESGModels.hamilton_step(Y[1,:], X[1,:], [Β1, Β2], [Σ1, Σ2], transition_matrix, states_zero )

        expected = [.763, .237]
        @test result[1] ≈ expected[1] atol=0.001
        @test result[2] ≈ expected[2] atol=0.001
        
    end
  
    @testset "hamilton filter" begin

        result = ESGModels.hamilton_filter(Y, X, [Β1, Β2], [Σ1, Σ2], transition_matrix, states_zero )

        @test size(result) == (2,2)
        expected_t1 = ESGModels.hamilton_step(Y[1,:], X[1,:], [Β1, Β2], [Σ1, Σ2], transition_matrix, states_zero)
        @test result[1,1] ≈ expected_t1[1]
        @test result[1,2] ≈ expected_t1[2]

        expected_t2 = ESGModels.hamilton_step(Y[2,:], X[2,:], [Β1, Β2], [Σ1, Σ2], transition_matrix, [.763, .237])
        @test result[2,1] ≈ expected_t2[1] atol=.001
        @test result[2,2] ≈ expected_t2[2] atol=.001
        
    end   
     
       
end 


@testset "Smoother" begin

    St = [.9, .1]    
    St1T = [.85, .15]

    transition_matrix = [.9 .1;.2 .8]
    
    result = smooth_step(St1T, St, transition_matrix)
    
    @test result ≈ [.909, .091] atol=0.001

    regime_probs = [.8 .2; .7 .3 ; .65 .35]

    result = smoother(regime_probs, transition_matrix)

    @test size(result) == (3,2)
    @test result[end,:] == [.65, .35]
    @test result[end-1,:] == smooth_step([.65, .35], [.7, .3], transition_matrix)
   

end

@testset "regime_params" begin
    
    Y = [.16 .05; .13 .045; .07 .03; .08 .035; .11 .04]

    X = [1. .12 .06; 1. .16 .05; 1. .13 .045; 1. .07 .03; 1. .11 .04]

    regimes_probs = [.9 .1; .8 .2; .7 .3; .7 .3; .8 .2]

    @test calc_xhat(X, regime_probs) ≈ [0.9487 0.1138 0.0569;
                                        0.8944 0.1431 0.0447;
                                        0.8367 0.1088 0.0376;
                                        0.8367 0.0586 0.0251;
                                        0.8944 0.0984 0.0358]

    #@test calc_regime_coeficientents(Y, X, regime_probs) ≈ [0.0196873 0.0283806; -0.187316 -0.0301025; 2.83093 0.458406]
end
