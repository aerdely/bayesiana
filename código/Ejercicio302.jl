### Ejercicio de Selección de Modelos
### Autor: Dr. Arturo Erdely
### Versión: 2025-05-09


## Cargar código externo necesario:

using Distributions
include("02probestim.jl") # en misma carpeta



## Simular muestras aleatorias

begin
    n = 50 # tamaño de muestras
    α, β = 3, 5 # parámetros 
    X = Gamma(α, β)
    muestra1 = rand(X, n)
    μ, σ = 2.7, 1.5 # parámetros
    Y = LogNormal(μ, σ)
    muestra2 = rand(Y, n)
end;

X
mean(X), var(X)
mean(muestra1), var(muestra1)
Y
mean(Y), var(Y)
mean(muestra2), var(muestra2)


## Método ABC 

function ABC(muestra::Vector{<:Real}, priori1::Matrix{<:Real}, priori2::Matrix{<:Real}, nsim = 1_000_000, nselec = 100)
    # priori1: matriz 2×2, cols -> valores mínimo y máximo, filas -> cada parámetro
    # priori2: lo análogo a priori1
    αsim = rand(Uniform(priori1[1,1], priori1[1,2]), nsim)
    βsim = rand(Uniform(priori1[2,1], priori1[2,2]), nsim)
    μsim = rand(Uniform(priori2[1,1], priori2[1,2]), nsim)
    σsim = rand(Uniform(priori2[2,1], priori2[2,2]), nsim)
    modelo = rand([1,2], nsim)
    δ = zeros(nsim)
    nn = length(muestra)
    muestraord = sort(muestra)
    for i ∈ 1:nsim
        if modelo[i] == 1 
            muestrasim = rand(Gamma(αsim[i], βsim[i]), nn)
        else
            muestrasim = rand(LogNormal(μsim[i], σsim[i]), nn)
        end
        δ[i] = sum(abs.(sort(muestrasim) - muestraord))
    end
    iselec = sortperm(δ)[1:nselec]
    p = masaprob(modelo[iselec])
    return p
end


## Experimentos 

priori1 = [0 20; 0 20]
priori2 = [-10 10; 0 20]

@time p1 = ABC(muestra1[1:20], priori1, priori2);
[[1,2] p1.fmp.([1,2])]

@time p2 = ABC(muestra2[1:20], priori1, priori2);
[[1,2] p2.fmp.([1,2])]

# Análisis secuencial 

m = n
probMuestra1 = zeros(m)
probMuestra2 = zeros(m)
@time for i ∈ 1:m 
    probMuestra1[i] = ABC(muestra1[1:i], priori1, priori2).fmp(1)
    probMuestra2[i] = ABC(muestra2[1:i], priori1, priori2).fmp(1)
end; # m=50 --> 2 minutos aprox

println(probMuestra1) # converge a 1.0
println(probMuestra2) # converge a 0.0

