using Statistics

sigma_tau = .0002 
sigma_g = 0 #.0006 
rho_c = .9700
theta_c = .4976
sigma_c = .0272 

n_samples = 2000
n_states = 4
n_steps = 100
st = zeros(n_samples, n_steps+1, n_states)
x = zeros(n_samples, n_steps+1)

A = [1 1 0 0
     0 1 0 0
     0 0 rho_c theta_c
     0 0 0 0]

C = [sigma_tau 0 0
     0 sigma_g 0
     0 0 sigma_c
     0 0 sigma_c]

G = [1, 0, 1, 0]

transition(A,C, previous_state) = A * previous_state .+ C*randn(3)
measurement(G, state) = G' * state   

st[:,1,:] .= [2.78 0.0115 0 0] # dodać średnią 10 letnią dividendę w 2000 r. dla trendu, i c roznica miedzy srednia a aktualna dywidendom

for s in 1:n_samples
    for t in 2:n_steps
        st[s,t,:] = transition(A,C,st[s,t-1,:])
        x[s,t,:] .= measurement(G,st[s,t,:])
    end
end

quantile(exp.(x[:,100]), [.05, .25, .5, .75, .95])

#gt mean .0577
#log(16.17)

#.0577 / 5

