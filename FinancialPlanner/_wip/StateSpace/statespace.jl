using StateSpaceModels, TimeSeries, Dates, CSV, Plots

pwd()

data_source = TimeArray(CSV.File("./FinancialPlanner/data/usa_var_data.csv"; delim=';', decimal=',',types =[DateTime, Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64], dateformat="dd.mm.yyyy"), timestamp=:Date)
data_source = collapse(data_source, Dates.quarter, last)
cpi = percentchange(data_source[:CPI], :log)
demeaned_cpi = cpi .- mean(cpi)

model = LocalLevel(randn(100))

fit!(model)

print_results(model)

ks = kalman_smoother(model)

plot(model, ks)