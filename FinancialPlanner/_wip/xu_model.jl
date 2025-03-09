
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

struct πu_params
    mean
end

struct πd_params
    mean
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
    σϕd
  
end


function inflation_step(π_params:: π_params, πu_params:: πu_params, πd_params:: πd_params, Θ_mean:: Float32, Θu_mean:: Float32, Θd_mean:: Float32, Θt::Float32, Θut::Float32, Θdt::Float32, πt::Float32, πut:: Float32, πdt:: Float32)

    ωπu = Gamma(πut, 1) - πut
    ωπd = Gamma(πdt, 1) = πdt
    
    Θ_term = π_params.pΘ * (Θt - Θ_mean) + π_params.pΘu * (Θut - Θu_mean)  + π_params.pΘd * (Θdt - Θd_mean) 
    π_term = π_params.pπ * (πt - π_params.mean) + π_params.pπu(πut - πu_params.mean) + π_params.pπd(πdt - πd_params.mean)
    u = (π_params.σΘu * ωΘu + π_params.σΘd * ωΘd) + (π_params.σπu * ωπu - π_params.σπd * ωπd)

    πt1 = π_params.mean + Θ_term + π_term + u
       
    return πt1    
        
end
