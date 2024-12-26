
module Bonds

export bond_return, scenarios_bond_returns

duration(yeld,M) = 1/yeld * (1- 1/((1+ 0.5 * yeld) ^ (2*M)))
convexity(yeld,M) = (2/yeld^2) * (1- 1/((1+ 0.5*yeld) ^ (2*M))) - (2*M)/(yeld*(1+0.5*yeld)^(2*M+1))

"""
    ```
    bond_return(yeldt, yeldtm1, M, dt) 
    ```
    
    calclulate single period bond return approximation based on yeld
    yeldt - yeld for current t
    yeldtm1 - yeld for previous t
    M - maturity
    dt - delta time
"""
function bond_return(yeldt:: Float64, yeldtm1:: Float64, M:: Int64, dt:: Int64)

    D = duration(yeldt, M)
    C = convexity(yeldt, M)
    
    return yeldtm1/dt - D * (yeldt - yeldtm1) + 0.5 * C * (yeldt - yeldtm1)^2

end

function scenarios_bond_returns(yeld_scenarios:: Matrix{Float64}, M:: Int, dt:: Int)
    n_scenarios, T = size(yeld_scenarios)
    result = zeros(n_scenarios, T-1)

    for s in 1:n_scenarios
        for t in 2:T
            result[s,t-1] = bond_return(yeld_scenarios[s,t], yeld_scenarios[s,t-1], M, dt)
        end
    end


    return result 
end


end
