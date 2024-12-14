#=
    Instalar algunos paquetes necesarios para el curso sobre Estadística Bayesiana.
=#

using Pkg # cargando el instalador de paquetes de Julia

begin
    paquete = ["QuadGK", "HCubature", "Optim",
               "SpecialFunctions", "LaTeXStrings", 
               "Distributions", "StatsBase",
               "Plots", "StatsPlots", "CSV", "DataFrames"
    ]
    for p in paquete
        println("*** Instalando paquete: ", p)
        Pkg.add(p)
    end
    println("*** Fin de lista de paquetes.")
end
