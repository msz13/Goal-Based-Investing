using LinearAlgebra, Distributions

function create_state_cov(ρ, B1, B2, σD, σg, σu, ρuD, ρuG, ρgD, ρM, σM)

    σμD = ρuD * σu * σD
    σμg = ρuG * σu * σg
    σgD = ρgD * σg * σD

    σr(ρ, B1, B2, σD, σg, σu, σμD, σμg)  = sqrt(σD^2 + ρ^2*B1^2 *σu^2 + ρ^2*B2^2*σg^2 − 2*ρ*B1*σμD − 2ρ^2*B1*B2*σμg)
    ΒM(ρM, σM, σr) = (ρM*σM)/σr
    σgM(ρ, ΒM, B1, B2, σgμ, σg) = −ΒM*ρ*B1*σgμ + ΒM*ρ*B2*σg^2
    σuM(ρ, ΒM, B1, B2, σgμ, σuD, σu) = ΒM*σuD -  ΒM*ρ*B1*σu^2 + ΒM*B2*ρ*σgμ
    σDM(ρ, ΒM, σD, σuD) = ΒM*σD^2 − ΒM*ρ*B1*σuD

    eσr =  σr(ρ, B1, B2, σD, σg, σu, σμD, σμg)

    eΒM =  ΒM(ρM, σM, eσr)

    eσgM = σgM(ρ, eΒM, B1, B2, σμg, σg)

    eσμM = σuM(ρ, eΒM, B1, B2, σμg, σμD, σu)

    eσDM = σDM(ρ, eΒM, σD, σμD)

    result = [σg^2 σμg σgD eσgM
            σμg σu^2 σμD eσμM
            σgD σμD σD^2 eσDM 
            eσgM eσμM eσDM σM^2 
    ]
    return result
end

function create_R()
    return [zeros(4)'
            I(4)
            zeros(4)'
            ]
end

function create_transition_matrix(γ1)
    return [γ1 0 1 0 0 0
            zeros(4,6)
            0 0 0 0 1 0
            ]
end

create_M0(γ0, δ1, A) = [γ0
                        (1-δ1)*A]

create_M1(δ1) = [0 0; 0 δ1]

create_M2(γ1, δ1, B1, B2) = [
                             1 1 0 0 1 -1
                             B2*(γ1 - δ1) 0 B2 -B1 -1 δ1
                            ]


struct KroijenModel
    F
    R
    Σ
    M0
    M1
    M2 
    function KroijenModel(ρ::Float64,
        A::Float64,
        B1::Float64,
        B2::Float64,    
        γ0::Float64,
        γ1::Float64,
        δ1::Float64,    
        σD::Float64,
        σg::Float64,
        σu::Float64,
        ρuD::Float64,
        ρuG::Float64,
        ρgD::Float64,    
        ρM::Float64,
        σM::Float64)

        F = create_transition_matrix(γ1)
        R = create_R()
        Σ = create_state_cov(ρ, B1, B2, σD, σg, σu, ρuD, ρuG, ρgD, ρM, σM)
        
        M0 = create_M0(γ0, δ1, A)        
        M1 =  create_M1(δ1)
        M2 = create_M2(γ1, δ1, B1, B2)

        return new(F, R, Σ, M0, M1, M2)
        
    end
end


sim_state_step(model, previous_state) = model.F* previous_state + model.R * rand(MvNormal(zeros(4), model.Σ))

sim_obs_step(model, state, previous_obs) = model.M0 + model.M1*previous_obs + model.M2* state



