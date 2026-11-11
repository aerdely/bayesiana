### Ejercicio de Selección de Modelos
### Autor: Dr. Arturo Erdely
### Versión: 2026-11-05


## Paquetes y funciones auxiliares

using Distributions, SpecialFunctions, Plots, LaTeXStrings # previamente instalados

include("02probestim.jl")


function ν(xx::Vector{<:Real}; α = 0.0001, β = 0.0001, δ = 0.0001, ε = 0.0001)
    # ν = P(muestra|M₂) / P(muestra|M₁)
    n = length(xx)
    s = sum(xx)
    slf = sum(lfactorial.(xx))
    logν = lbeta(δ+n,ε+s) + lgamma(α) + slf + (α+s)*log(β+n) - α*log(β) - lgamma(α+s) - lbeta(δ,ε)
    return exp(logν)
end

function probM₁(muestra::Vector{<:Real})
    n = length(muestra)
    p = zeros(n)
    for i ∈ 1:n 
        p[i] = 1 / (1 + ν(muestra[1:i]))
    end
    return p
end


## Experimento

begin
    n = 100 # tamaño de muestra
    θ = 0.38
    λ = (1-θ)/θ
    M₁ = Poisson(λ)
    M₂ = Geometric(θ)
    xxM₁ = rand(M₁, n)
    xxM₂ = rand(M₂, n)
end;

M₁ # Poisson
mean(M₁), var(M₁) # media y varianza teóricas
mean(xxM₁), var(xxM₁) # media y varianza muestrales
M₂ # Geométrica
mean(M₂), var(M₂) # media y varianza teóricas
mean(xxM₂), var(xxM₂) # media y varianza muestrales

# con muestra Poisson
begin
  pM₁ = probM₁(xxM₁); # probabilidades secuenciales
  plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_1))", legend = false)
end

# con muestra Geométrica
begin
  pM₁ = probM₁(xxM₂); # probabilidades secuenciales
  plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_2))", legend = false)
end

# con muestra mezclada al 50%
begin # repetir varias veces
  n = 100 # probar con distintos valores
  b = rand([0,1], n)
  xxM₁ = rand(M₁, n)
  xxM₂ = rand(M₂, n)
  xx = b .* xxM₁ + (1 .- b) .* xxM₂
  pM₁ = probM₁(xx); # probabilidades secuenciales
  plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(mezcla))", legend = false)
end


## Método ABC 

function métodoABC(xx::Vector{<:Real}; nsims = 1_000_000, nselec = 300)
    n = length(xx)
    ss(x) = sort(x)
    ssobs = ss(xx)
    dist(x,y) = sum(abs.(x - y))
    λsim = rand(Uniform(0, 3*mean(xx)), nsims)
    θsim = rand(Uniform(0,1), nsims)
    bsim = rand([0,1], nsims)
    d = zeros(nsims)
    for i in 1:nsims
	xx1sim = rand(Poisson(λsim[i]), n)
	xx2sim = rand(Geometric(θsim[i]), n)
	xxsim = bsim[i] .* xx1sim .+ (1 - bsim[i]) .* xx2sim
	d[i] = dist(ss(xxsim), ssobs)
    end
    is = sortperm(d)[1:nselec]
    sb = sum(bsim[is])
    va = Beta(1/2 + sb, 1/2 + nselec - sb)
    estint = intervalo(rand(va, 100_000))
    return(med = median(va), I = estint, va = va)
end


# Experimento
begin
    n = 100 # tamaño de muestra
    θ = 0.38
    λ = (1-θ)/θ
    M₁ = Poisson(λ)
    M₂ = Geometric(θ)
    xxM₁ = rand(M₁, n)
    xxM₂ = rand(M₂, n)
end;

# con muestra Poisson
pM₁ = probM₁(xxM₁)
k = 6 # probar distintos valores de k ∈ {1,...,n}
pM₁[k]
@time abc = métodoABC(xxM₁[1:k], nsims = 10_000_000)


# con muestra Geométrica
pM₁ = probM₁(xxM₂)
k = 11 # probar distintos valores de k ∈ {1,...,n}
pM₁[k]
@time abc = métodoABC(xxM₂[1:k], nsims = 10_000_000)
