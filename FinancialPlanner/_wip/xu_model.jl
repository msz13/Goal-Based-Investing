
struct Output_Params
    mean
    ρθ
    ρθu
    ρθd
    σΘu
    σΘd
end

struct OutputState_Params
    u_mean
    pΘu
    d_mean
    pΘd
    σΘu
    σΘd
end


function output_step(Output_Params, OutputState_Params, Θt0, Θut, Θdt)
    
    mean, ρΘΘ, ρΘθu, ρθθd, σΘθu, σΘΘd  = Output_Params.mean, Output_Params.ρΘ, Output_Params.ρθu, Output_Params.ρθd, Output_Params.σΘu, Output_Params.σΘd
    

    #= ωΘu = rand(Gamma(abs(Θut - Θu_mean),1))
    ωΘd = rand(Gamma(abs(Θdt - Θd_mean),1)) =#
    ωΘu = rand(Gamma(Θut)) - Θut
    ωΘd = rand(Gamma(Θdt)) - Θdt
    uθ = σΘΘu * ωΘu - σΘΘd * ωΘd 
    
    Θt1 = Θ_mean + ρθθ*(Θt - Θ_mean) + ρθθu*(Θut - Θu_mean) + ρθθd*(Θdt - Θd_mean) + uθ 
    Θut1 = Θu_mean + pΘu*(Θut - Θu_mean) + σΘu * ωΘu
    Θdt1 = Θd_mean + pΘd*(Θdt - Θd_mean) + σΘd * ωΘd

    return Θt1, Θut1, Θdt1, ωΘu, ωΘd
end

struct Output_Params
    mean
    ρθ
    ρθu
    ρθd
    σΘu
    σΘd
end

struct OutputState_Params
    u_mean
    pΘu
    d_mean
    pΘd
    σΘu
    σΘd
end

function output_simulation(Output_Params, OutputState_Params, Θt0, Θut0,  Θdt0, n_steps)
    Θ = zeros(n_steps)
    Θu = zeros(n_steps)
    Θd = zeros(n_steps)

    Θ[1], Θu[1], Θd[1] = output_step(Output_Params, OutputState_Params, Θt0, Θut0, Θdt0 )

   #=  for t in 1:n_steps
        Θ, Θu, Θd = output_step()

    end =#
end

struct Θ_params
    Θ_mean
    Θu_mean
    Θd_mean
end

struct πstate_params
    πu_mean
    πd_mean
end


struct π_params
    mean
    pΘ
    pΘu
    pΘd
    pπ
    pπu
    pπd
    σΘu
    σΘd
    σπu
    σπd
  
end



function inflation_step(π:: π_params, π_state:: πstate_params, Θ_mean:: Float64, Θu_mean:: Float64, Θd_mean:: Float64, πt:: Float64, πut:: Float64, πdt:: Float64, Θt::Float64, Θut::Float64, Θdt::Float64, ωΘu:: Float64, ωΘd:: Float64)

    ωπu = rand(Gamma(πut, 1)) - πut
    ωπd = rand(Gamma(πdt, 1)) - πdt
    
    Θ_term = π.pΘ * (Θt - Θ_mean) + π.pΘu * (Θut - Θu_mean)  + π.pΘd * (Θdt - Θd_mean) 
    π_term = π.pπ * (πt - π.mean) + π.pπu * (πut - π_state.πu_mean) + π.pπd * (πdt - π_state.πd_mean)
    u = (π.σΘu * ωΘu + π.σΘd * ωΘd) + (π.σπu * ωπu - π.σπd * ωπd)

    πt1 = π.mean + Θ_term + π_term + u
       
    return πt1    
        
end 
 
#= 
function inflation_step(π:: π_params, Θ_mean:: Float64, Θu_mean:: Float64, Θt:: Float64, Θut:: Float64)
    
    Θ_term = π.pΘ * (Θt - Θ_mean) + π.pΘu * (Θut - Θu_mean)

    πt1 = π.mean + Θ_term
       
    return πt1    
    
end  =#