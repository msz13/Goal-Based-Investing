
function local_level_ARMA()

    A = [1 1 0 0
     0 1 0 0
     0 0 rho_c theta_c
     0 0 0 0]

    C = [sigma_tau 0 0
     0 sigma_g 0
     0 0 sigma_c
     0 0 sigma_c]

    G = [1, 0, 1, 0]

    return StateSpaceModel(T, R, Z, Q, H, initial_state_mean, initial_state_covariance)
    
end