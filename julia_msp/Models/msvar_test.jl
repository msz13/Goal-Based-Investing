using Revise
using Test, StatsBase

includet("msvar.jl")
using .MSVAR

@testset "Next regime" begin
    result = MSVAR.next_regime([.67, .33], [.9 .1;.2 .8])
    @test result ≈ [0.669, 0.331] atol=0.001
end


@testset "Pdfs" begin

    y = [.16, .05]

    X = [1, .12, .06]

    
    Β1 = [.08 .3 .2; .03 .1 .6]'
    Σ1 = cor2cov([1 .3; .3 1], [.15, .03])
    Β2 = [.085 .35 .25; .035 .15 .63]'
    Σ2 = cor2cov([1 .35; .35 1], [.16, .035])

    result = MSVAR.likehood_t(y, X, [Β1, Β2], [Σ1, Σ2])

    @test size(result) == (2,)
    @test result[1] ≈ 20.98 atol=0.01 
    @test result[2] ≈ 13.18 atol=0.01 

end

@testset "Hamilton_step" begin

    y = [.16, .05]

    X = [1., .12, .06]

    transition_matrix = [.9 .1;.2 .8]

    states_zero = [.67, .33]

    
    Β1 = [.08 .3 .2; .03 .1 .6]'
    Σ1 = cor2cov([1 .3; .3 1], [.15, .03])
    Β2 = [.085 .35 .25; .035 .15 .63]'
    Σ2 = cor2cov([1 .35; .35 1], [.16, .035])
    
    expected = [.763, .237]

    result = MSVAR.hamilton_step(y, X, [Β1, Β2], [Σ1, Σ2], transition_matrix, states_zero )

    @test result[1] ≈ expected[1] atol=0.001
    @test result[2] ≈ expected[2] atol=0.001
    
end


@testset "Hamilton_filter" begin

    Y = [.16 .05; .13 .045]

    X = [1. .12 .06; 1. .16 .05]

    transition_matrix = [.9 .1;.2 .8]

    states_zero = [.67, .33]

    
    Β1 = [.08 .3 .2; .03 .1 .6]'
    Σ1 = cor2cov([1 .3; .3 1], [.15, .03])
    Β2 = [.085 .35 .25; .035 .15 .63]'
    Σ2 = cor2cov([1 .35; .35 1], [.16, .035])

        
    result = hamilton_filter(Y, X, [Β1, Β2], [Σ1, Σ2], transition_matrix, states_zero )

    @test size(result) == (2,2)
    expected_t1 = MSVAR.hamilton_step(Y[1,:], X[1,:], [Β1, Β2], [Σ1, Σ2], transition_matrix, states_zero)
    @test result[1,1] ≈ expected_t1[1]
    @test result[1,2] ≈ expected_t1[2]
    expected_t2 = MSVAR.hamilton_step(Y[2,:], X[2,:], [Β1, Β2], [Σ1, Σ2], transition_matrix, [.763, .237])
    @test result[2,1] ≈ expected_t2[1] atol=.001
    @test result[2,2] ≈ expected_t2[2] atol=.001
   
       
end 

