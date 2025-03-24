
# zrobić reprezentację matrycowa dla modelu inflacji
# zasymulować inflacje dla modelu z parametrami z papersa dla 4 kwartalow
# zrobic w state space estymymacje, local level i sesonal
# symulacja dla 4 kwartalow
# zrobic state space estymacje dla moje go modelu
# symulacja dla 4 kwartalow

using Distributions, LinearAlgebra

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

quantile(π_year, [.02, .25, .5, .75, .98]) ./ 10

#drugie parametry

d = 1.04

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

quantile(y, [.02, .25, .5, .75, .98]) .* 4

π_year = sum(y[:,3:42], dims=2)

quantile(π_year, [.02, .25, .5, .75, .98]) ./ 10

#equity premium

d = 0

H = 2.21*10^-9
Q = [4.83*10^-2 -1.13; -1.13 41.05]

Θ1 = 0.333
Θ2 = 0.338

α0 = [0, 0, 0]

T = [1 0 0; 0 Θ1 Θ2; 0 1 0]
Z = [1, 1, 0]

n_steps = 42 
n_samples = 10000

y, states = sample_pi(Z, H, d, T, Q, α0, n_steps, n_samples)

quantile(y[:, 5], [.05, .25, .5, .75, .95]) .* 4

e_year = sum(y[:,3:42], dims=2)

quantile(e_year, [.02, .25, .5, .75, .98]) ./ 10


#full model

ep = 4.83*10^-2
rp = 2.03*10^-2
πp = 5.37*10^-2 
ea = 41.05 
ra = 4.95*10^-1
πa = 2.65*10^-1 
da = 2.07
τa = 1.92
eatm1 = 31.05 
ratm1 = 3.95*10^-1
πatm1 = 1.65*10^-1 
datm1 = 1.07
τatm1 = 0.92

St  = [ep, rp, πp, ea, ra, πa, da, τa, eatm1, ratm1, πatm1, datm1, τatm1]


Θ1 = [.333, .343, .003, .528, .234]
Θ2 = [.338, .286, -.130, -.012, .625]

F = [I(3)       zeros(3, 5) zeros(3, 5)
     zeros(5,3) diagm(Θ1)   diagm(Θ2)
     zeros(5,3) I(5)        zeros(5,5)
     ]

(F * St)[1:7]
(F * St)[8:13]

F[1:8,:]
F[9:13,:]



p_avg = 3.57 #3.37
est_rho(p_avg) = exp(p_avg)/(1+ exp(p_avg))

ρ =  est_rho(p_avg)
n = 40
H1 = [[0, -1, 0, 0, -1, 0, 1, 0]; zeros(5)]'
He = [[-1, 0, 0, -1, 0, 0, 0, 0]; zeros(5)]'
H2 = [[0, 1, 1, 0, 1, 1, 0, 0]; zeros(5)]'
Hτ = [[0, 0, 0, 0, 0, 0, 0, 1]; zeros(5)]'
H3 = [[1, 0, 0, 1, 0, 0, 0, 0,]; zeros(5)]'
H4 = [[0, 0, 0, 0, 0, 0, 1, 0,]; zeros(5)]'

1/n .* (H2' .* ((ones(13) - diag(F).^n) ./ (ones(13) - diag(F)) .* diag(F)))'

diag(F)'


acoeff = (ones(5)  - Θ1.^n) ./ (ones(5)  - Θ1) .* Θ1 * 1/n .* [0, 1, 1, 0, 0] + (ones(5)  - Θ1.^(n-1)) ./ (ones(5)  - Θ1) .* Θ1 * 1/n .* [0, 0, 0, 0, 1]

pcoeff = [0, 1, 1]

Hli = [pcoeff; acoeff; zeros(5)]'

Hi = [0, 1, 1, 0, (1/n) * (((1 - Θ1[2]^n)/(1 - Θ1[2])) * Θ1[2]), (1/n) * (((1 - Θ1[3]^n)/(1 - Θ1[3])) * Θ1[3]), 0, (1/n) * (((1 - Θ1[5]^(n-1))/(1 - Θ1[5])) * Θ1[5]), 0, (1/n) * (((1 - Θ2[2]^n)/(1 - Θ2[2])) * Θ2[2]), (1/n) * (((1 - Θ2[3]^n)/(1 - Θ2[3])) * Θ2[3]), 0, (1/n) * (((1 - Θ2[5]^(n-1))/(1 - Θ2[5])) * Θ2[5])]


H = [((H1 + He)' .* (diag(F) ./ (ones(13) - ρ * diag(F))));
       Hi
       H2 * F
       H3
       H4
    ]

H2

F[1:8,:]
F[9:13,:]

F[:,10]'

H2 * F[:,10] 

H2 * F

1/n * (H2 * (ones(13,13) - F.^n) * inv(ones(13,13) - F))
(Hτ * (ones(13,13) - F.^(n-1)) * inv(ones(13,13) - F)) + (Hτ * (ones(13,13) - F.^(n-1)) * inv(ones(13,13) - F))

1/n * (H2 * (ones(13,13) - F.^n) * inv(ones(13,13) - F))

(ones(13,13) - F.^n) * inv(ones(13,13) - F)

#= H = [(H1 + He) * inv(I(13) - ρ * F) * F
     H2 * F
     H3
     H4
    ] =#

A = [1 2 3; 4 5 6; 7 8 9]
Ht = [1 0 1]

H2
Ht * A 

F[:,10]'

H2 * F
H2 * F * F'

.003^2




coeff = [[1, 1, 1]; Θ1; Θ2]

H2 =  [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]

Hi = (H2 .* (((ones(13) - coeff.^n) ./ (ones(13) - coeff)) .* coeff) * (1/n))'

Hi = [0, 1, 1, 0, (1/n) * (((1 - Θ1[2]^n)/(1 - Θ1[2])) * Θ1[2]), (1/n) * (((1 - Θ1[3]^n)/(1 - Θ1[3])) * Θ1[3]), 0, (1/n) * (((1 - Θ1[5]^(n-1))/(1 - Θ1[5])) * Θ1[5]), 0, (1/n) * (((1 - Θ2[2]^n)/(1 - Θ2[2])) * Θ2[2]), (1/n) * (((1 - Θ2[3]^n)/(1 - Θ2[3])) * Θ2[3]), 0, (1/n) * (((1 - Θ2[5]^(n-1))/(1 - Θ2[5])) * Θ2[5])]'






