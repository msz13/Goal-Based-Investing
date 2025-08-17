


"""
State Space Model structure
x_t = F * x_{t-1} + G * u_t    (state equation)
y_t = H * x_t + v_t            (observation equation)

where:
- x_t is the state vector at time t
- y_t is the observation vector at time t
- u_t ~ N(0, Q) is the state noise
- v_t ~ N(0, H) is the observation noise
- T is the state transition matrix
- R is the state noise coefficient matrix
- Z is the observation matrix
"""

struct StateSpaceModel
    T::Matrix{Float64}  # State transition matrix
    R::Matrix{Float64}  # State noise coefficient matrix
    Z::Matrix{Float64}  # Observation matrix
    Q::Matrix{Float64}  # State noise covariance
    H::Matrix{Float64}  # Observation noise covariance
    initial_state_mean::Vector{Float64}
    initial_state_covariance::Matrix{Float64}


end


function tc_var(var_coeff, trend_cov, cycle_cov, initial_trend_mean, initial_cycle_mean, initial_trend_covariance, initial_cycle_covariance)

    T = [I(2)      zeros(2,2) # Transition  matrix
         zeros(2,2) var_coeff
         ]

    Q = [trend_cov zeros(2,2) #State noise covariance
         zeros(2,2) cycle_cov]

    R = Matrix(I, 4 , 4)  # State noise coefficient matrix
    Z = [1 0 1 0
         1 1 0 1]  # Observation matrix

    H = Matrix(I, 2, 2) * eps()  # Observation noise covariance

     initial_state_mean = [initial_trend_mean; initial_cycle_mean]

     initial_state_covariance = [initial_trend_covariance zeros(2,2)
                                 zeros(2,2) initial_cycle_covariance]

  
    return StateSpaceModel(T, R, Z, Q, H, initial_state_mean, initial_state_covariance)
    

end


function sample(model:: StateSpaceModel, n_steps)

    n_variables, n_states = size(model.Z)
    states = zeros(n_steps, n_states)
    obs = zeros(n_steps, n_variables)

    states[1, :] = rand(MvNormal(model.initial_state_mean, model.initial_state_covariance))
    obs[1, :] = model.Z * states[1,:] .+ rand(MvNormal(zeros(n_variables), model.H))
    
    for t in 2:n_steps
        states[t,:] = model.T * states[t-1,:] + rand(MvNormal(zeros(n_states), model.Q))
        obs[t, :] = model.Z * states[t,:] + rand(MvNormal(zeros(n_variables), model.H))
    end

    return states, obs

end