using StateSpaceModels, TimeSeries, Dates, CSV, Plots, StatsBase

pwd()

data_source = TimeArray(CSV.File("./FinancialPlanner/data/usa_var_data.csv"; delim=';', decimal=',',types =[DateTime, Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64], dateformat="dd.mm.yyyy"), timestamp=:Date)
data_source = collapse(data_source, Dates.quarter, last)
cpi = percentchange(data_source[:CPI], :log)
demeaned_cpi = cpi .- mean(values(cpi))



mean(cpi) .* 4

plot(values(cpi))
plot(values(demeaned_cpi))

#model = UnobservedComponents(values(demeaned_cpi); trend = "local level")
model = BasicStructural(values(demeaned_cpi),2)


StateSpaceModels.fit!(model)

print_results(model)

ks = kalman_smoother(model)

forec = forecast(model, 20)

plot(model, ks)

plot(model, forec)

scenarios = simulate_scenarios(model, 40, 1000)

quantile(sum(scenarios, dims=1)[1,1,:], [.05, .25, .5, .75, .95]) ./10 .+ .034

sqrt.(model.system.Q) .* 4^0.5

sqrt.(model.system.H) * 4^0.5

model.system.R