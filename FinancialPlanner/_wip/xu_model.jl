
function output_step(Θ_mean, ρθθ, ρθθu, ρθθd, Θt, Θu_mean, pΘu, Θut, Θd_mean, pΘd, Θdt, σΘΘu, σΘΘd, σΘu, σΘd)
    
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