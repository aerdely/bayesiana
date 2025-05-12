### Ejercicio de Selección de Modelos
### Autor: Dr. Arturo Erdely
### Versión: 2025-05-08


## Cargar paquetes y código externo necesarios:

using Distributions, Plots, LaTeXStrings # previamente instalados
include("zExponencial.jl")


## Funciones

function ν(xx::Vector{<:Real})
    # ν = P(muestra|M₂) / P(muestra|M₁)
    m = length(xx)
    s = sum(xx)
    slog = sum(log.(xx))
    logν = m*log(slog) + slog - m*log(s - m)
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
    n = 1_000 # tamaño de muestra
    α = 5.0 
    θ = 4.0
    M₁ = Pareto(α, 1.0)
    M₂ = vaExponencial(1.0, θ)
    xxM₁ = rand(M₁, n)
    xxM₂ = M₂.sim(n)
end;
M₁
mean(M₁), var(M₁)
mean(xxM₁), var(xxM₁) # Pareto
M₂.familia, M₂.param
M₂.media, M₂.varianza
mean(xxM₂), var(xxM₂) # Exponencial

# con muestra Pareto
pM₁ = probM₁(xxM₁) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_1))", legend = false)

# con muestra Exponencial
pM₁ = probM₁(xxM₂) # probabilidades secuenciales
plot(pM₁, lw = 2, xlabel = L"n", title = L"P(M_1\,|\,\mathbf{x}_{obs}(M_2))", legend = false)

