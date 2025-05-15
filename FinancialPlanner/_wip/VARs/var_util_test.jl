using Test

include("turing_nw_bvar.jl")

@testset "prepare_var_data Tests" begin

    # Test case 1: Basic VAR data preparation (no exogenous variables, no intercept)
    Y = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    p1 = 1
    Y_obs1, predictors1 = prepare_var_data(Y, p1)
    @test size(Y_obs1) == (3, 2)
    @test size(predictors1) == (3, 2)
    @test predictors1[1, :] == [1.0, 2.0]
    @test predictors1[2, :] == [3.0, 4.0]
    @test predictors1[3, :] == [5.0, 6.0]
    @test Y_obs1[1, :] == [3.0, 4.0]
    @test Y_obs1[2, :] == [5.0, 6.0]
    @test Y_obs1[3, :] == [7.0, 8.0] 
   

    #Test case 2: VAR data preparation with 2 lags
    p2 = 2
    Y_obs2, predictors2 = prepare_var_data(Y, p2)
    @test size(Y_obs2) == (2, 2)
    @test size(predictors2) == (2, 4)
    @test predictors2[1, :] == [1.0, 2.0, 3.0, 4.0]
    @test predictors2[2, :] == [3.0, 4.0, 5.0, 6.0]   
    @test Y_obs2[1, :] == [5.0, 6.0]
    @test Y_obs2[2, :] == [7.0, 8.0]
   

   # Test case 3: VAR data preparation with exogenous variables
    X3 = [0.1, 0.2, 0.3, 0.4]
    p3 = 1
    Y_obs3, predictors3 = prepare_var_data(Y, p3, X3)
    @test size(Y_obs3) == (3, 2)
    @test size(predictors3) == (3, 3)
    @test predictors3[1, :] == [1.0, 2.0, 0.2]
    @test predictors3[2, :] == [3.0, 4.0, 0.3]
    @test predictors3[3, :] == [5.0, 6.0, 0.4]
    @test Y_obs1[1, :] == [3.0, 4.0]
    @test Y_obs1[2, :] == [5.0, 6.0]
    @test Y_obs1[3, :] == [7.0, 8.0]  
    
    
    # Test case 4: VAR data preparation with intercept
    p4 = 1
    Y_obs4, predictors4 = prepare_var_data(Y, p4, Matrix{Float64}(undef, 0, 0), true)
    @test size(Y_obs4) == (3, 2)
    @test size(predictors4) == (3, 3)
    @test predictors4[1, :] == [1.0, 1.0, 2.0]
    @test predictors4[2, :] == [1.0, 3.0, 4.0]
    @test predictors4[3, :] == [1.0, 5.0, 6.0]
    

    # Test case 5: VAR data preparation with exogenous variables and intercept
    X5 = [0.1 0.2; 0.3 0.4; 0.5 0.6; 0.7 0.8] 
    p5 = 1
    Y_obs5, predictors5 = prepare_var_data(Y, p5, X5, true)
    @test size(Y_obs5) == (3, 2)
    @test size(predictors5) == (3, 5)
    @test predictors5[1, :] == [1.0, 1.0, 2.0, 0.3, 0.4]
    @test predictors5[2, :] == [1.0, 3.0, 4.0, 0.5, 0.6]
    @test predictors5[3, :] == [1.0, 5.0, 6.0, 0.7, 0.8]
   
   

    # Test case 6: Error if X has incorrect number of rows
    X6 = [0.1, 0.2]
    p6 = 1
    @test_throws ErrorException prepare_var_data(Y, p6, X6) 
end

runtests(tests="prepare_var_data Tests")