
# zrobić reprezentację matrycowa dla modelu inflacji
# zasymulować inflacje dla modelu z parametrami z papersa dla 4 kwartalow
# zrobic w state space estymymacje, local level i sesonal
# symulacja dla 4 kwartalow
# zrobic state space estymacje dla moje go modelu
# symulacja dla 4 kwartalow

using Distributions, Statistics

α0 #initial states
Z # state coeff
T # transition state matrix
H # measurement std
Q # states covariance matrix
d # measurement constant term

f(d, Z, state, H) = d + Z' * state + rand(Normal(0, H)) # measurement equtation
gs(T, Q, statetm1) = T * statetm1 + [rand(MvNormal(zeros(2), Q)); 0] # state equtation

d = 0.55

H = 3.01 * 10^-13
Q = [5.51 * 10^-2 8.84*10^-2 ; 8.84*10^-2 2.53*10^-1]

Θ1 = -0.006 
Θ2 = -0.140

α0 = [0, 0, 0]

T = [1 0 0; 0 Θ1 Θ2; 0 1 0]
Z = [1, 1, 0]

n_steps = 42 
n_samples = 1000

function sample_pi(Z, H, d, T, Q, state0, n_steps, n_samples)
    states = zeros(n_samples, n_steps, 3)
    y = zeros(n_samples, n_steps)

    for s in 1:n_samples
        states[s, 1,:] = gs(T, Q, α0)
        y[s, 1] = f(d, Z, states[s, 1,:], H)
        for t in 2:n_steps
            states[s, t, :] = gs(T, Q, states[s, t-1, :])
            y[s, t] = f(d, Z, states[s, t, :], H)
        end
    end

    return y, states
end

y, states = sample_pi(Z, H, d, T, Q, α0, n_steps, n_samples)


π_year = sum(y[:,3:42], dims=2)

quantile(π_year, [.02, .25, .5, .75, .95]) ./ 10

#drugie parametry

d = 0.55

H = 3.51 * 10^-14 
Q = [5.37 * 10^-2 -9.69*10^-2 ; -9.69*10^-2 2.65*10^-1]

Θ1 = 0.003
Θ2 = -0.130

α0 = [0, 0, 0]

T = [1 0 0; 0 Θ1 Θ2; 0 1 0]
Z = [1, 1, 0]

n_steps = 42 
n_samples = 10000

y, states = sample_pi(Z, H, d, T, Q, α0, n_steps, n_samples)


π_year = sum(y[:,3:42], dims=2)

quantile(π_year, [.02, .25, .5, .75, .95]) ./ 10


