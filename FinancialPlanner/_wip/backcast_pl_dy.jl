import Pkg
Pkg.activate("./FinancialPlanner")

using DataFrames, TimeSeries, XLSX, CSV, StatsBase, Plots, GLM

include(".//VARs/utils.jl")


df = DataFrame(XLSX.readtable("./FinancialPlanner/data/poland_equity.xlsx", "poland", infer_eltypes=true))
#data_source = collapse(TimeArray(df; timestamp = :Date), Dates.quarter, last)

v = Matrix(df[:,2:end])
t = collect(Date(1998,12,31):Month(1):Date(2025,4,30))
n = names(df[!,2:end])

data_source = collapse(TimeArray(t, v, n), Dates.year, last)
data_source = from(data_source, Date(1999, 01, 01))

returns = percentchange(data_source[[:msci_net, :msci_gross, :msci_price, :mwi40, :mwig40tr]], :log) 


mean(returns[[:msci_net, :msci_gross, :msci_price, :mwi40]])


D = expm1.(returns[:mwig40tr] .-  returns[:mwi40]) .* data_source[:mwi40] .* exp.(-returns[:mwi40] ./ 2)
mdy = D ./ data_source[:mwi40] 
mdy = mdy[Date(2010,01,01):Date(2024,12,31)] .* 100
mean(mdy)
mean(from(mwig_dy, Date(2010,01,01)))

plot(mdy)
plot!(from(mwig_dy, Date(2010,01,01)))
plot!(twinx(), from(returns[:mwi40], Date(2010,01,01)); color=:orange)

#msci_dy = expm1.(returns[:msci_gross] .- returns[:msci_net ]) 
msci_dy = returns[:msci_gross] .- returns[:msci_price] 
mean(msci_dy)

s_msci_dy = msci_dy[Date(2001,01,01):Date(2006,12,31)] 
l_msci_dy = msci_dy[Date(2007,01,01):Date(2024,12,31)] 
mwig_dy = data_source[:mwig40_dy][Date(2007,01,01):Date(2024,12,31)] 
mwig_dy = TimeArray(timestamp(mwig_dy), Float64.(values(mwig_dy)), colnames(mwig_dy))

mean(msci_dy)
mean(s_msci_dy)
mean(l_msci_dy)
mean(mwig_dy)




plot(l_msci_dy)
plot!(mwig_dy)

returns[:mwi40][Date(2007,01,01):Date(2024,12,31)]

long_seriers = merge(l_msci_dy, mwig_dy, )

ols = lm(@formula(mwig40_dy ~ msci_gross_msci_price), long_seriers)

r2(ols)

predicted = predict(ols, s_msci_dy)
dy_pred = TimeArray(timestamp(s_msci_dy),predicted, [:mwig_dy])

plot(s_msci_dy)
plot!(dy_pred)



xtix = collect(Date(1999, 12, 31):Year(5):Date(2024,12,31))

plot(data_source[:msci_price])
plot!(twinx(), msci_dy; color=:orange)



plot(data_source[:mwi40]; size=(800,600), xticks=7, xformatter = :auto, date_format = "yyyy-mm-dd")
plot!(twinx(), data_source[:mwig40_dy]; color=:red)

head(msci_dy,8)
head(returns[:mwi40],9)

D = msci_dy .* data_source[:msci_price]
d = percentchange(D, :log)
head(d, 8)
head(D, 8)


plot(d)

d[Date(2011,01,01):Date(2019,12,31)]