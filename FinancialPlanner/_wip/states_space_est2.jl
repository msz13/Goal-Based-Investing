using Distributions, LinearAlgebra

#full model

ep = Θ1[1] * 0.5 * 4.83 * 10^-2 + Θ1[2] * 0.5 * 4.83 * 10^-2
rp = 2.03 * 10^-2
πp = 5.37 * 10^-2
ea = 41.05
ra = 4.95 * 10^-1
πa = 2.65 * 10^-1
da = 2.07 + Θ1[4] * 2.07
τa = 1.92
eatm1 = 31.05
ratm1 = 3.95 * 10^-1
πatm1 = 1.65 * 10^-1
datm1 = 1.07
τatm1 = 1.92

St = [ep, rp, πp, ea, ra, πa, da, τa, eatm1, ratm1, πatm1, datm1, τatm1]

Θ1[4] / (1 - ρ * Θ1[4]) * da - rp - ep

Θ1 = [0.333, 0.343, 0.003, 0.528, 0.234]
Θ2 = [0.338, 0.286, -0.130, -0.012, 0.625]

p_avg = 3.57 #3.37
est_rho(p_avg) = exp(p_avg) / (1 + exp(p_avg))


ρ = est_rho(p_avg)
n = 40

F = [I(3) zeros(3, 5) zeros(3, 5)
    zeros(5, 3) diagm(Θ1) diagm(Θ2)
    zeros(5, 3) I(5) zeros(5, 5)
]



St1 = F * St
St1'

H1 = [[0, -1, 0, 0, -1, 0, 1, 0]; zeros(5)]'
He = [[-1, 0, 0, -1, 0, 0, 0, 0]; zeros(5)]'
H2 = [[0, 1, 1, 0, 1, 1, 0, 0]; zeros(5)]'
Hτ = [[0, 0, 0, 0, 0, 0, 0, 1]; zeros(5)]'
H3 = [[1, 0, 0, 1, 0, 0, 0, 0,]; zeros(5)]'
H4 = [[0, 0, 0, 0, 0, 0, 1, 0,]; zeros(5)]'



H = [
    (H1 + He) * inv(I(13) - ρ * F) * F
    (1 / n) * (H2 * (I(13) - F^n) * (I(13) - F) * F + Hτ * (I(13) - F^(n - 1)) * (I(13) - F) * F)
    H2 * F
    H3
    H4
]

Yt = H * St ./ [1, 100, 100, 100, 100]

Yt'

Yt1 = H * St1 ./ [1, 100, 100, 100, 100]

Yt1'


# Full model with persistent dividends


vdp = 4.66 * 10^-2
vrp = 2.34 * 10^-2
vπp = 5.51 * 10^-2
vda = 2.2
vra = 5.0 * 10^-1
vπa = 2.53 * 10^-1
vea = 62.111
vτa = 2.09


dp = sqrt(vdp)
rp = sqrt(vrp)
πp = sqrt(vπp)
da = sqrt(vda)
ra = sqrt(vra)
πa = sqrt(vπa)
ea = sqrt(vea)
τa = sqrt(vτa)
dptm1 = sqrt(vdp)
rptm1 = sqrt(vrp)
πptm1 = sqrt(vπp)
datm1 = sqrt(vda)
ratm1 = sqrt(vra)
πatm1 = sqrt(vπa)
eatm1 = sqrt(vea)
τatm1 = sqrt(vτa)

St = [ep, rp, πp, ea, ra, πa, da, τa, eatm1, ratm1, πatm1, datm1, τatm1]

Θ1 = [0.532, 0.339, -0.006, 0.269, 0.222]
Θ2 = [-0.020, 0.286, -0.140, 0.318, 0.630]

p_avg = 3.57 #3.37
est_rho(p_avg) = exp(p_avg) / (1 + exp(p_avg))


ρ = est_rho(p_avg)
n = 40

p = 1 / (1 - ρ) * dp + Θ1[1] / (1 - ρ * Θ1[1]) * da - 1 / (1 - ρ) * rp - Θ1[4] / (1 - ρ * Θ1[4]) * ea


F = [I(3) zeros(3, 5) zeros(3, 5)
    zeros(5, 3) diagm(Θ1) diagm(Θ2)
    zeros(5, 3) I(5) zeros(5, 5)
]

St1 = F * St
St1'

H1 = [[1, -1, 0, 1, -1, 0, 0, 0]; zeros(5)]'
He = [[0, 0, 0, 0, 0, 0, 1, 0]; zeros(5)]'
H2 = [[0, 1, 1, 0, 1, 1, 0, 0]; zeros(5)]'
Hτ = [[0, 0, 0, 0, 0, 0, 0, 1]; zeros(5)]'
H3 = [[1, 0, 0, 1, 0, 0, 0, 0]; zeros(5)]'
H4 = [[0, 0, 1, 0, 0, 1, 0, 0]; zeros(5)]'

H = [
    (H1 + He) * inv(I(13) - ρ * F) * F
    (1 / n) * ((H2 * (I(13) - F^n) * (I(13) - F) * F + Hτ * (I(13) - F^(n - 1)) * (I(13) - F) * F))
    H2 * F
    H3
    H4
] 

H * St1


Hp = [1/(1-ρ), -1/(1-ρ), 0, Θ1[1]/(1- ρ*Θ1[1]), -Θ1[2]/(1- ρ*Θ1[2]), 0, -Θ1[4]/(1- ρ*Θ1[4]), 0, 0, 0, 0, 0, 0]'
Hin = [0, 1, 1, 0, (1/n)*((1 - Θ1[2]^n)/(1 - Θ1[2]))*Θ1[2], (1/n)*((1 -Θ1[3]^n)/(1 .-Θ1[3]))*Θ1[3], 0, (1/n)*((1 -Θ1[5]^(n-1))/(1 .-Θ1[5]))*Θ1[5], 0, 0, 0, 0, 0]'
Hi = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]'
Hd = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]'
Hπ = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]'

H = [Hp 
     Hin
     Hi 
     Hd 
     Hπ
     ]

V = []

f(F, V, Stm1) = F * Stm1 + MvNormal(zeros(13, V))
     
Yt = H * St
Yt'

