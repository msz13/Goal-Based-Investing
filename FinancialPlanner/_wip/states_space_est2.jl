using Distributions, LinearAlgebra, StatsBase

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
vea = 62.11
vτa = 2.09
vrp_dp = 2.99*10^-2
vπp_dp = -3.55*10^-2
vda_dp = -1.21*10^-1
vra_dp = -4.53*10^-2
vπa_dp = 2.52*10^-2
vea_dp = 1.25
vτa_dp = -4.99^10-3
vπp_rp = -1.95*10^-2
vda_rp = -3.38*10^-2
vra_rp = -4.56*106-2
vπa_rp = -1.65*10^-3
vea_rp =  5.8*10^-1
vτa_rp = -2.76*10^-2
vda_πp = 1.86*10^-1
vra_πp = -7.67*10-3
vπa_πp = -8.84*10^-2
vea_πp = -1.09
vτa_πp = -2.32*10^-1
vra_da = -3.47*10^-1
vπa_da = -3.78*10^-1
vea_da = -3.71
vτa_da = -9.23*10^-1
vπa_ra =  2.49*10^-1
vea_ra = -2.72*10^-1
vτa_ra = 9.48*10^-1
vea_πa = 1.49
vτa_πa = 6.28*10^-1
vτa_ea = 1.2


V = [vdp vrp_dp vπp_dp vda_dp vra_dp vπa_dp vea_dp vτa_dp zeros(5)'
     vrp_dp vrp vπp_rp vda_rp vra_rp vπa_rp vea_rp vτa_rp zeros(5)'    
     vπp_dp vπp_rp vπp vda_πp vra_πp vπa_πp vea_πp vτa_πp zeros(5)'
     vda_dp vda_rp vda_πp vda vra_da vπa_da vea_da vτa_da zeros(5)'
     vra_dp vra_rp vra_πp vra_da vra vπa_ra vea_ra vτa_ra zeros(5)'
     vπa_dp vπa_rp vπa_πp vπa_da vπa_ra vπa vea_πa vτa_πa zeros(5)'
     vea_dp vea_rp vea_πp vea_da vea_ra vea_πa vea vτa_ea zeros(5)'
     vτa_dp vτa_rp vτa_πp vτa_da vτa_ra vτa_πa vτa_ea vτa zeros(5)'
     zeros(5,13)
]

isposdef(V)
isposdef(Hermitian(V))

Symmetric(V)
ispossemdef(M::Matrix{Fl}) where Fl = all(i -> i >= 0.0, eigvals(M))
ispossemdef(V)

sigmas = sqrt.([vdp, vrp, vπp, vda, vra, vπa, vea, vτa]) 

CorrM = Symmetric([1 .91 -.7 -.38 -.3 .23 .74 -0.02
         0 1 -.54 -.15 -.42 -.0 .48 -.12
         0 0 1 .53 -.46 -.75 -59 -.68
         0 0 0 1 -.31 -.51 -.32 -.43
         0 0 0 0 1 .7 -.05 .93
         0 0 0 0 0 1 .37 .86
         0 0 0 0 0 0 1 .11
         0 0 0 0 0 0 0 1
])

CorrM = collect(CorrM)

isposdef(CorrM)

VC = cor2cov(CorrM, sigmas)

isposdef(VC)
issymmetric(VC)

eigen(VC).values

MvNormal(zeros(8), Hermitian(VC))

R = diagm([1.45*10^-1, 1.59*10^-14, 3.01*10^-13, 7.43*10^-10, 3.78*10^-12 ])



#= dp = sqrt(vdp)
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
=#

#St = [ep, rp, πp, ea, ra, πa, da, τa, eatm1, ratm1, πatm1, datm1, τatm1]



Θ1 = [0.532, 0.339, -0.006, 0.269, 0.222]
Θ2 = [-0.020, 0.286, -0.140, 0.318, 0.630]

p_avg = 3.37 #3.57 
est_rho(p_avg) = exp(p_avg) / (1 + exp(p_avg))
ρ = est_rho(p_avg)

n = 40

F = [I(3) zeros(3, 5) zeros(3, 5)
    zeros(5, 3) diagm(Θ1) diagm(Θ2)
    zeros(5, 3) I(5) zeros(5, 5)
]

H1 = [[1, -1, 0, 1, -1, 0, 0, 0]; zeros(5)]'
He = [[0, 0, 0, 0, 0, 0, 1, 0]; zeros(5)]'
H2 = [[0, 1, 1, 0, 1, 1, 0, 0]; zeros(5)]'
Hτ = [[0, 0, 0, 0, 0, 0, 0, 1]; zeros(5)]'
H3 = [[1, 0, 0, 1, 0, 0, 0, 0]; zeros(5)]'
H4 = [[0, 0, 1, 0, 0, 1, 0, 0]; zeros(5)]'


#= H = [
    (H1 + He) * inv(I(13) - ρ * F) * F
    (1 / n) * ((H2 * (I(13) - F^n) * (I(13) - F) * F + Hτ * (I(13) - F^(n - 1)) * (I(13) - F) * F))
    H2 * F
    H3
    H4
]  =#



Hp = [1/(1-ρ), -1/(1-ρ), 0, Θ1[1]/(1- ρ*Θ1[1]), -Θ1[2]/(1- ρ*Θ1[2]), 0, -Θ1[4]/(1- ρ*Θ1[4]), 0, Θ2[1]/(1- ρ*Θ2[1]), -Θ2[2]/(1- ρ*Θ2[2]), 0, -Θ2[4]/(1- ρ*Θ2[4]), 0]'
Hin = [0, 1, 1, 0, (1/n)*((1 - Θ1[2]^n)/(1 - Θ1[2]))*Θ1[2], (1/n)*((1 -Θ1[3]^n)/(1 .-Θ1[3]))*Θ1[3], 0, (1/n)*((1 -Θ1[5]^(n-1))/(1 .-Θ1[5]))*Θ1[5], 0, (1/n)*((1 - Θ2[2]^n)/(1 - Θ2[2]))*Θ2[2], (1/n)*((1 -Θ2[3]^n)/(1 .-Θ2[3]))*Θ2[3], 0, (1/n)*((1 -Θ2[5]^(n-1))/(1 .-Θ2[5]))*Θ2[5]]'
Hi = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]'
Hd = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]'
Hπ = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]'

H = [Hp 
     Hin
     Hi 
     Hd 
     Hπ
     ]


f(F, V, Stm1) = rand(MvNormal(F * Stm1, Hermitian(V)))
g(H, St, R) = rand(MvNormal(H * St, diagm(R)))

T = 22
n_samples = 1000

St0 = zeros(13)

states = zeros(13, n_samples, T) 

states[:, 1, 1] = f(F, VC, St0)

