import Pkg
Pkg.add("ScenTrees")

Pkg.add("EconomicScenarioGenerators")
Pkg.add("FinanceModels")
Pkg.add("Plots")
Pkg.add("PyPlot")

using EconomicScenarioGenerators
using FinanceModels
using Plots
using ScenTrees

function gbm()
    m = BlackScholesMerton(0.07,0.00,0.15,1.0)
    s = ScenarioGenerator(
               1,  # timestep
               2, # projection horizon
               m,  # model
           )
    return collect(s)
end


path = gbm()

plot(0:2,path)

cagr = path[end] ^ (1/2) -1

tree = Tree([1,2,2],1)

tree_plot(tree)

tree_approximation!(tree,gbm,100000,2,2)

tree.name
tree.state
tree.probability

tree_plot(tree)

for s in tree.state
    println(s)
end

for s in tree.probability
    println(s)
end

for s in tree.parent
    println(s)
end

tree.children

tree.parent

tree.state[5]

tree.probability[7]

stage(tree)

root(tree,7)

leaves(tree)

tree.state[tree.parent[5]]

tree.state[5]

i =2
plot!(stage(tree)[i],[tree.state[tree.parent[i]],tree.state[i]])

i =3
plot!(stage(tree)[i],[tree.state[tree.parent[i]],tree.state[i]])

plot!(1:2,1:2)

