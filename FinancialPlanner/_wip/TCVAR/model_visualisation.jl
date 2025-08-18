
function plot_variable_states(observations, states, titles)
    
    data = hcat(observations, states)

    plot(data; layout=(3,1), size=(800, 600), title=titles)

end

