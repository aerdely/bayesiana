### Ejemplo de Selección de Modelos
### Autor: Dr. Arturo Erdely
### Versión: 2025-05-02


## Cargar paquetes y código externo necesarios:

using Distributions, Plots, LaTeXStrings, SpecialFunctions # previamente instalados



## Funciones

function ν(xx::Vector{<:Real})
    # ν = P(muestra|M₂) / P(muestra|M₁)
    m = length(xx)
    s = sum(xx)
    z = 0.0
    for j ∈ 1:m
        if xx[j] ≥ 1
            for r ∈ 1:xx[j]
                z += log(r)
            end
        end
    end
    logν = lgamma(m) + (s + 1/2)*log(m) + z - lgamma(m + s - 1/2)
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


## Experimento 1

begin
    n = 100 # tamaño de muestra
    λ = 5 
    θ = 0.4
    M₁ = Poisson(λ)
    M₂ = Geometric(θ)
    xxM₁ = rand(M₁, n)
    xxM₂ = rand(M₂, n)
end;
M₁
mean(M₁), var(M₁)
mean(xxM₁), var(xxM₁) # Poisson
M₂
mean(M₂), var(M₂)
mean(xxM₂), var(xxM₂) # Geométrica

# con muestra Poisson 
pM₁ = probM₁(xxM₁) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_1))", legend = false)

# con muestra Geométrica
pM₁ = probM₁(xxM₂) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_2))", legend = false)


## Experimento 2

begin
    n = 100 # tamaño de muestra
    λ = 5 
    θ = 1 / (1 + λ)
    M₁ = Poisson(λ)
    M₂ = Geometric(θ)
    xxM₁ = rand(M₁, n)
    xxM₂ = rand(M₂, n)
end;
M₁
mean(M₁), var(M₁)
mean(xxM₁), var(xxM₁) 
M₂
mean(M₂), var(M₂)
mean(xxM₂), var(xxM₂) # Geométrica

# con muestra Poisson 
pM₁ = probM₁(xxM₁) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_1))", legend = false)

# con muestra Geométrica
pM₁ = probM₁(xxM₂) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_2))", legend = false)


## Experimento 3

begin
    n = 1000 # tamaño de muestra: probar con 100 y 10,000
    λ = 0.1 
    θ = 1 / (1 + λ)
    M₁ = Poisson(λ)
    M₂ = Geometric(θ)
    xxM₁ = rand(M₁, n)
    xxM₂ = rand(M₂, n)
end;
M₁
mean(M₁), var(M₁)
mean(xxM₁), var(xxM₁) 
M₂
mean(M₂), var(M₂)
mean(xxM₂), var(xxM₂) # Geométrica

# con muestra Poisson 
pM₁ = probM₁(xxM₁) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_1))", legend = false)

# con muestra Geométrica
pM₁ = probM₁(xxM₂) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_2))", legend = false)

