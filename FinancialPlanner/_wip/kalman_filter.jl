
# Kalman Filter for multivariate state-space model
function kalman_filter(y, A, C, Q, R, x0, P0, u=nothing)
    # y: observations (T × n matrix, n = observation dimension)
    # A: state transition matrix (m × m, m = state dimension)
    # C: observation matrix (n × m)
    # Q: state noise covariance (m × m)
    # R: observation noise covariance (n × n)
    # x0: initial state mean (m × 1)
    # P0: initial state covariance (m × m)
    # u: optional exogenous inputs (T × p matrix, p = input dimension)
    
    T, n = size(y)
    m = size(A, 1)
    p = isnothing(u) ? 0 : size(u, 2)
    
    # Storage for results
    x_pred = zeros(T, m)  # Predicted states
    P_pred = zeros(T, m, m)  # Predicted covariances
    x_filt = zeros(T, m)  # Filtered states
    P_filt = zeros(T, m, m)  # Filtered covariances
    loglik = 0.0  # Log-likelihood
    
    x_t = x0  # Initial state
    P_t = P0  # Initial covariance
    
    for t in 1:T
        # Prediction step
        x_pred[t, :] = A * x_t
        if !isnothing(u)
            x_pred[t, :] += B * u[t, :]  # Include exogenous inputs if provided
        end
        P_pred[t, :, :] = A * P_t * A' + Q
        
        # Update step
        y_t = y[t, :]
        y_pred = C * x_pred[t, :]

        if !isnothing(u)
            y_pred += D * u[t, :]  # Include exogenous inputs if provided
        end
        innov = y_t - y_pred  # Innovation
        S = C * P_pred[t, :, :] * C' + R  # Innovation covariance
        K = P_pred[t, :, :] * C' * inv(S)  # Kalman gain
        
        x_filt[t, :] = x_pred[t, :] + K * innov
        P_filt[t, :, :] = P_pred[t, :, :] - K * C * P_pred[t, :, :]
        
        # Update log-likelihood
        loglik += logpdf(MvNormal(y_pred, Hermitian(S)), y_t)
        
        # Update state and covariance for next iteration
        x_t = x_filt[t, :]
        P_t = P_filt[t, :, :]
    end
    
    return x_filt, P_filt, x_pred, P_pred, loglik
end

# Carter-Kohn Smoother for multivariate state-space model
function carter_kohn_smoother(x_filt, P_filt, x_pred, P_pred, A, Q)
    # x_filt: filtered states from Kalman filter (T × m)
    # P_filt: filtered covariances (T × m × m)
    # x_pred: predicted states (T × m)
    # P_pred: predicted covariances (T × m × m)
    # A: state transition matrix (m × m)
    # Q: state noise covariance (m × m)
    
    T, m = size(x_filt)
    
    # Storage for smoothed states and covariances
    x_smooth = zeros(T, m)
    P_smooth = zeros(T, m, m)
    
    # Initialize with last filtered state
    x_smooth[T, :] = x_filt[T, :]
    P_smooth[T, :, :] = P_filt[T, :, :]
    
    # Backward pass
    for t in T-1:-1:1
        # Smoother gain
        J = P_filt[t, :, :] * A' * inv(P_pred[t+1, :, :])
        
        # Smoothed state
        x_smooth[t, :] = x_filt[t, :] + J * (x_smooth[t+1, :] - x_pred[t+1, :])
        
        # Smoothed covariance
        P_smooth[t, :, :] = P_filt[t, :, :] + J * (P_smooth[t+1, :, :] - P_pred[t+1, :, :]) * J'
    end
    
    return x_smooth, P_smooth
end