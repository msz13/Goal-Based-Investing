using Test
using LinearAlgebra
using Distributions
using Random
using Statistics

# Include the main implementation (assuming it's in a separate file)
# include("carter_kohn_algorithm.jl")

# Test data generation functions
function generate_simple_ar1_model()
    """Generate a simple AR(1) state space model for testing"""
    F = [0.8]
    G = [1.0]
    H = [1.0]
    Q = [1.0]
    R = [0.5]
    initial_state_mean = [0.0]
    initial_state_covariance = [2.0]
    
    return StateSpaceModel(F, G, H, Q, R, initial_state_mean, initial_state_covariance)
end

function generate_multivariate_model()
    """Generate a 2D state space model for testing"""
    F = [0.9 0.1; 0.0 0.8]
    G = [1.0 0.0; 0.0 1.0]
    H = [1.0 0.0; 0.0 1.0]
    Q = [1.0 0.2; 0.2 1.0]
    R = [0.5 0.0; 0.0 0.8]
    initial_state_mean = [0.0, 0.0]
    initial_state_covariance = [2.0 0.0; 0.0 2.0]
    
    return StateSpaceModel(F, G, H, Q, R, initial_state_mean, initial_state_covariance)
end

function simulate_data(model::StateSpaceModel, n_time_steps::Int; seed::Int=123)
    """Simulate data from a state space model"""
    Random.seed!(seed)
    n_states = length(model.initial_state_mean)
    n_obs = size(model.H, 1)
    
    true_states = zeros(n_time_steps, n_states)
    observations = zeros(n_time_steps, n_obs)
    
    # Initial state
    true_states[1, :] = rand(MvNormal(model.initial_state_mean, model.initial_state_covariance))
    observations[1, :] = model.H * true_states[1, :] + rand(MvNormal(zeros(n_obs), model.R))
    
    # Simulate forward
    for t in 2:n_time_steps
        state_noise = rand(MvNormal(zeros(size(model.Q, 1)), model.Q))
        true_states[t, :] = model.F * true_states[t-1, :] + model.G * state_noise
        
        obs_noise = rand(MvNormal(zeros(n_obs), model.R))
        observations[t, :] = model.H * true_states[t, :] + obs_noise
    end
    
    return true_states, observations
end

@testset "Kalman Filter Tests" begin
    
    @testset "Basic Functionality - AR(1) Model" begin
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 50)
        
        state_filtered, covariance_filtered, state_predicted, covariance_predicted, log_likelihood = 
            kalman_filter(model, observations)
        
        # Test dimensions
        @test size(state_filtered) == (50, 1)
        @test size(covariance_filtered) == (50, 1, 1)
        @test size(state_predicted) == (50, 1)
        @test size(covariance_predicted) == (50, 1, 1)
        
        # Test that covariances are positive definite
        for t in 1:50
            @test covariance_filtered[t, 1, 1] > 0
            @test covariance_predicted[t, 1, 1] > 0
        end
        
        # Test that log-likelihood is finite
        @test isfinite(log_likelihood)
    end
    
    @testset "Multivariate Model" begin
        model = generate_multivariate_model()
        true_states, observations = simulate_data(model, 30)
        
        state_filtered, covariance_filtered, state_predicted, covariance_predicted, log_likelihood = 
            kalman_filter(model, observations)
        
        # Test dimensions
        @test size(state_filtered) == (30, 2)
        @test size(covariance_filtered) == (30, 2, 2)
        
        # Test that covariance matrices are positive definite
        for t in 1:30
            cov_filtered_t = reshape(covariance_filtered[t, :, :], 2, 2)
            cov_predicted_t = reshape(covariance_predicted[t, :, :], 2, 2)
            
            @test isposdef(cov_filtered_t + 1e-10*I)
            @test isposdef(cov_predicted_t + 1e-10*I)
        end
    end
    
    @testset "Perfect Observations (R = 0)" begin
        # Test with very small observation noise (near-perfect observations)
        model = generate_simple_ar1_model()
        model_perfect = StateSpaceModel(
            model.F, model.G, model.H, model.Q, [1e-10],
            model.initial_state_mean, model.initial_state_covariance
        )
        
        true_states, observations = simulate_data(model_perfect, 20)
        state_filtered, covariance_filtered, _, _, _ = kalman_filter(model_perfect, observations)
        
        # With perfect observations, filtered states should be very close to observations
        for t in 1:20
            @test abs(state_filtered[t, 1] - observations[t, 1]) < 1e-8
            @test covariance_filtered[t, 1, 1] < 1e-8
        end
    end
    
    @testset "Missing Observations" begin
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 20)
        
        # Introduce missing observations
        observations[5:7, 1] .= NaN
        observations[15, 1] = NaN
        
        state_filtered, covariance_filtered, state_predicted, covariance_predicted, _ = 
            kalman_filter(model, observations)
        
        # For missing observations, filtered should equal predicted
        for t in [5, 6, 7, 15]
            @test state_filtered[t, 1] ≈ state_predicted[t, 1]
            @test covariance_filtered[t, 1, 1] ≈ covariance_predicted[t, 1, 1]
        end
    end
    
    @testset "Steady State Behavior" begin
        # For stable systems, covariances should converge
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 200)
        
        _, covariance_filtered, _, covariance_predicted, _ = kalman_filter(model, observations)
        
        # Check convergence in later time steps
        late_filtered_vars = [covariance_filtered[t, 1, 1] for t in 180:200]
        late_predicted_vars = [covariance_predicted[t, 1, 1] for t in 180:200]
        
        # Variance should be relatively stable in steady state
        @test std(late_filtered_vars) < 0.1 * mean(late_filtered_vars)
        @test std(late_predicted_vars) < 0.1 * mean(late_predicted_vars)
    end
end

@testset "Carter-Kohn Smoother Tests" begin
    
    @testset "Basic Functionality - AR(1) Model" begin
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 30)
        
        # Test with small number of samples for speed
        state_smoothed_samples = carter_kohn_sampler(model, observations, n_samples=50)
        
        # Test dimensions
        @test size(state_smoothed_samples) == (50, 30, 1)
        
        # Test that all samples are finite
        @test all(isfinite.(state_smoothed_samples))
        
        # Compute posterior statistics
        state_smoothed_mean, state_smoothed_lower, state_smoothed_upper = 
            compute_posterior_statistics(state_smoothed_samples)
        
        @test size(state_smoothed_mean) == (30, 1)
        @test size(state_smoothed_lower) == (30, 1)
        @test size(state_smoothed_upper) == (30, 1)
        
        # Credible intervals should be ordered correctly
        @test all(state_smoothed_lower .<= state_smoothed_mean)
        @test all(state_smoothed_mean .<= state_smoothed_upper)
    end
    
    @testset "Multivariate Model" begin
        model = generate_multivariate_model()
        true_states, observations = simulate_data(model, 20)
        
        state_smoothed_samples = carter_kohn_sampler(model, observations, n_samples=30)
        
        # Test dimensions
        @test size(state_smoothed_samples) == (30, 20, 2)
        @test all(isfinite.(state_smoothed_samples))
        
        # Test posterior statistics
        state_smoothed_mean, _, _ = compute_posterior_statistics(state_smoothed_samples)
        @test size(state_smoothed_mean) == (20, 2)
    end
    
    @testset "Consistency with Kalman Filter" begin
        # At the final time point, smoothed mean should be close to filtered mean
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 25)
        
        # Run Kalman filter
        state_filtered, _, _, _, _ = kalman_filter(model, observations)
        
        # Run Carter-Kohn sampler
        state_smoothed_samples = carter_kohn_sampler(model, observations, n_samples=1000)
        state_smoothed_mean, _, _ = compute_posterior_statistics(state_smoothed_samples)
        
        # At final time point, smoothed and filtered should be very similar
        final_time_diff = abs(state_smoothed_mean[end, 1] - state_filtered[end, 1])
        @test final_time_diff < 0.1  # Should be close but not identical due to sampling
    end
    
    @testset "Smoothing Effect" begin
        # Smoothed estimates should generally have lower variance than filtered
        model = generate_simple_ar1_model()
        
        # Create data with some noise
        true_states, observations = simulate_data(model, 40)
        
        # Run both algorithms
        state_filtered, _, _, _, _ = kalman_filter(model, observations)
        state_smoothed_samples = carter_kohn_sampler(model, observations, n_samples=500)
        state_smoothed_mean, _, _ = compute_posterior_statistics(state_smoothed_samples)
        
        # Compute empirical variances (excluding boundary effects)
        mid_range = 10:30  # Avoid boundary effects
        
        filtered_innovations = diff(state_filtered[mid_range, 1])
        smoothed_innovations = diff(state_smoothed_mean[mid_range, 1])
        
        # Smoothed path should generally be... smoother (lower innovation variance)
        # This is a statistical test, so we use a generous tolerance
        @test var(smoothed_innovations) <= 1.5 * var(filtered_innovations)
    end
    
    @testset "Reproducibility" begin
        # Same seed should give same results
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 15, seed=456)
        
        Random.seed!(789)
        samples1 = carter_kohn_sampler(model, observations, n_samples=10)
        
        Random.seed!(789)
        samples2 = carter_kohn_sampler(model, observations, n_samples=10)
        
        @test samples1 ≈ samples2
    end
    
    @testset "Sample Properties" begin
        # Test that samples have approximately correct empirical properties
        model = generate_simple_ar1_model()
        
        # Use a simple, controlled case
        n_time_steps = 50
        true_states, observations = simulate_data(model, n_time_steps, seed=999)
        
        state_smoothed_samples = carter_kohn_sampler(model, observations, n_samples=2000)
        
        # Check that sample mean converges (law of large numbers)
        sample_means = mean(state_smoothed_samples, dims=1)[1, :, 1]
        
        # The empirical variance across samples should be reasonable
        for t in 1:n_time_steps
            sample_var = var(state_smoothed_samples[:, t, 1])
            @test sample_var > 0  # Should have positive variance
            @test sample_var < 10  # But not unreasonably large
        end
    end
    
    @testset "Edge Cases" begin
        model = generate_simple_ar1_model()
        
        # Test with minimal data
        minimal_obs = randn(2, 1)
        samples_minimal = carter_kohn_sampler(model, minimal_obs, n_samples=10)
        @test size(samples_minimal) == (10, 2, 1)
        @test all(isfinite.(samples_minimal))
        
        # Test with single time point
        single_obs = randn(1, 1)
        samples_single = carter_kohn_sampler(model, single_obs, n_samples=10)
        @test size(samples_single) == (10, 1, 1)
        @test all(isfinite.(samples_single))
    end
end

@testset "Integration Tests" begin
    
    @testset "End-to-End Workflow" begin
        # Test complete workflow from model specification to results
        model = generate_multivariate_model()
        true_states, observations = simulate_data(model, 35, seed=555)
        
        # Step 1: Kalman filtering
        state_filtered, covariance_filtered, state_predicted, covariance_predicted, log_likelihood = 
            kalman_filter(model, observations)
        
        @test isfinite(log_likelihood)
        
        # Step 2: Carter-Kohn sampling
        state_smoothed_samples = carter_kohn_sampler(model, observations, n_samples=100)
        
        # Step 3: Posterior statistics
        state_smoothed_mean, state_smoothed_lower, state_smoothed_upper = 
            compute_posterior_statistics(state_smoothed_samples, credible_level=0.90)
        
        # Verify complete workflow produces sensible results
        @test all(isfinite.(state_smoothed_mean))
        @test all(isfinite.(state_smoothed_lower))
        @test all(isfinite.(state_smoothed_upper))
        
        # Check coverage (true states should often fall within credible intervals)
        # This is probabilistic, so we check that most (not all) are covered
        n_covered = sum((true_states[:, 1] .>= state_smoothed_lower[:, 1]) .& 
                       (true_states[:, 1] .<= state_smoothed_upper[:, 1]))
        coverage_rate = n_covered / size(true_states, 1)
        
        # Should cover roughly 90% (allowing for sampling variability)
        @test coverage_rate > 0.75  # Generous bound for small sample
    end
    
    @testset "Performance Characteristics" begin
        # Test that algorithms complete in reasonable time and don't blow up
        model = generate_simple_ar1_model()
        true_states, observations = simulate_data(model, 100)
        
        # Time the Kalman filter (should be fast)
        time_kf = @elapsed begin
            kalman_filter(model, observations)
        end
        
        # Time the Carter-Kohn sampler (will be slower)
        time_ck = @elapsed begin
            carter_kohn_sampler(model, observations, n_samples=100)
        end
        
        @test time_kf < 1.0  # Kalman filter should be very fast
        @test time_ck < 10.0  # Carter-Kohn should complete reasonably quickly
        
        # Test memory usage doesn't explode
        samples = carter_kohn_sampler(model, observations, n_samples=50)
        @test sizeof(samples) < 1e8  # Should use less than ~100MB
    end
end

# Function to run all tests
function run_all_tests()
    println("Running comprehensive tests for Kalman Filter and Carter-Kohn Algorithm...")
    println("=" ^ 70)
    
    # Run the test suite
    Test.run_tests()
    
    println("=" ^ 70)
    println("All tests completed!")
end

# Uncomment to run tests
# run_all_tests()