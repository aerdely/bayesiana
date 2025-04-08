### Ejemplo muestreador de Gibbs
### Autor: Dr. Arturo Erdely
### Versión: 2025-04-04

#=
    Exponencial(μ,θ)  μ ∈ ℜ    θ > 0
=#


## Paquetes y código necesarios

using Distributions, Random, Plots, LaTeXStrings
include("xbExponencial.jl")


## Simular muestra y análisis conjugado no informativo

μ,θ = -1.5, 3.7; # valor teórico de los parámetros desconocidos
X = vaExponencial(μ,θ);
n = 30; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada

# modelo bayesiano no informativo:
post = bExpo(xx);
keys(post)

# muestreo sin reemplazo
function sinreemplazo(A::Array, m::Integer)
    # muestra sin reemplazo de tamaño m a partir de n elementos (m ≤ n)
    # requiere haber ejectado primero: using Random
    if m > length(A)
        error("m debe ser igual o menor que el número de elementos de A")
        return nothing
    else
        return shuffle(A)[1:m]
    end
end



## Muestreador de Gibbs 

begin # generar cadenas de Markov
    θG = 100.0 # valor inicial: θG > 0
    nsim = 100_000
    M = zeros(nsim)
    T = zeros(nsim)
    for k ∈ 1:nsim
        M[k] = post.rμ_θ(1, θG)[1]
        T[k] = post.rθ_μ(1, M[k])[1]
        θG = T[k]
    end
end

@show(μ); # valor teórico
begin # estimaciones puntuales de Gibbs
    M0 = M[1:10_000] # primeros 10 mil valores (no recomendable)
    M1 = M[90_001:100_000] # últimos 10 mil valores (recomendable)
    M2 = sinreemplazo(M[10_001:100_000], 10_000) # muestra sin reemplazo (muy recomendable)
    median.([M0, M1, M2])
end
post.qμ(0.5) # estimación puntual conjugada no informativa

@show(θ); # valor teórico 
begin # estimaciones puntuales de Gibbs
    T0 = T[1:10_000] # primeros 10 mil valores (no recomendable)
    T1 = T[90_001:100_000] # últimos 10 mil valores (recomendable)
    T2 = sinreemplazo(T[10_001:100_000], 10_000) # muestra sin reemplazo (muy recomendable)
    median.([T0, T1, T2])
end
post.qθ(0.5) # estimación puntual conjugada no informativa


## Densidades marginales a posteriori de μ y θ

begin
    histogram(M0, normalize = true, label = "Gibbs M0")
    t = range(post.qμ(0.001), post.qμ(0.999), length = 1_000)
    xaxis!(L"\mu"); yaxis!(L"p(\mu\,|\,\mathbf{x}_{obs})")
end 
histogram!(M1, normalize = true, label = "Gibbs M1")
histogram!(M2, normalize = true, label = "Gibbs M2")
plot!(t, post.dμ.(t), lw = 3, label = "a posteriori no info", color = :red)

begin
    histogram(T0, normalize = true, label = "Gibbs T0")
    t = range(post.qθ(0.001), post.qθ(0.999), length = 1_000)
    xaxis!(L"\theta"); yaxis!(L"p(\theta\,|\,\mathbf{x}_{obs})")
end 
histogram!(T1, normalize = true, label = "Gibbs T1")
histogram!(T2, normalize = true, label = "Gibbs T2")
plot!(t, post.dθ.(t), lw = 3, label = "a posteriori no info", color = :red)


## Análisis de las cadenas generadas 

begin
    plot(M[1:100], lw = 1, xlabel = L"k", ylabel = L"μ(k)", label = "")
    title!("Parte inicial: primeros 100 pasos")
    hline!([μ], lw = 2, color = :red, label = "valor teórico de μ = $μ")
end
begin
    plot(M[(end-99):end], lw = 1, xlabel = L"k", ylabel = L"μ(k)", label = "")
    title!("Parte final: últimos 100 pasos")
    hline!([μ], lw = 2, color = :red, label = "valor teórico de μ = $μ")
end

begin
    plot(T[1:100], lw = 1, xlabel = L"k", ylabel = L"θ(k)", label = "")
    title!("Parte inicial: primeros 100 pasos")
    hline!([θ], lw = 2, color = :red, label = "valor teórico de θ = $θ")
end
begin
    plot(T[(end-99):end], lw = 1, xlabel = L"k", ylabel = L"θ(k)", label = "")
    title!("Parte final: últimos 100 pasos")
    hline!([θ], lw = 2, color = :red, label = "valor teórico de θ = $θ")
end



## ¡Cuidado con el valor inicial!

begin # generar cadenas de Markov
    μG = -10.0 # valor inicial: μG ∈ ℜ # intenta μG = 0.0
    nsim = 100_000
    M = zeros(nsim)
    T = zeros(nsim)
    for k ∈ 1:nsim
        T[k] = post.rθ_μ(1, μG)[1]
        M[k] = post.rμ_θ(1, T[k])[1]
        μG = M[k]
    end
end

@show(μ); # valor teórico
begin # estimaciones puntuales de Gibbs
    M0 = M[1:10_000] # primeros 10 mil valores (no recomendable)
    M1 = M[90_001:100_000] # últimos 10 mil valores (recomendable)
    M2 = sinreemplazo(M[10_001:100_000], 10_000) # muestra sin reemplazo (muy recomendable)
    median.([M0, M1, M2])
end
post.qμ(0.5) # estimación puntual conjugada no informativa

@show(θ); # valor teórico 
begin # estimaciones puntuales de Gibbs
    T0 = T[1:10_000] # primeros 10 mil valores (no recomendable)
    T1 = T[90_001:100_000] # últimos 10 mil valores (recomendable)
    T2 = sinreemplazo(T[10_001:100_000], 10_000) # muestra sin reemplazo (muy recomendable)
    median.([T0, T1, T2])
end
post.qθ(0.5) # estimación puntual conjugada no informativa


## Densidades marginales a posteriori de μ y θ

begin
    histogram(M0, normalize = true, label = "Gibbs M0")
    t = range(post.qμ(0.001), post.qμ(0.999), length = 1_000)
    xaxis!(L"\mu"); yaxis!(L"p(\mu\,|\,\mathbf{x}_{obs})")
end 
histogram!(M1, normalize = true, label = "Gibbs M1")
histogram!(M2, normalize = true, label = "Gibbs M2")
plot!(t, post.dμ.(t), lw = 3, label = "a posteriori no info", color = :red)

begin
    histogram(T0, normalize = true, label = "Gibbs T0")
    t = range(post.qθ(0.001), post.qθ(0.999), length = 1_000)
    xaxis!(L"\theta"); yaxis!(L"p(\theta\,|\,\mathbf{x}_{obs})")
end 
histogram!(T1, normalize = true, label = "Gibbs T1")
histogram!(T2, normalize = true, label = "Gibbs T2")
plot!(t, post.dθ.(t), lw = 3, label = "a posteriori no info", color = :red)


## Análisis de las cadenas generadas 

begin
    plot(M[1:100], lw = 1, xlabel = L"k", ylabel = L"μ(k)", label = "")
    title!("Parte inicial: primeros 100 pasos")
    hline!([μ], lw = 2, color = :red, label = "valor teórico de μ = $μ")
end
begin
    plot(M[(end-99):end], lw = 1, xlabel = L"k", ylabel = L"μ(k)", label = "")
    title!("Parte final: últimos 100 pasos")
    hline!([μ], lw = 2, color = :red, label = "valor teórico de μ = $μ")
end

begin
    plot(T[1:100], lw = 1, xlabel = L"k", ylabel = L"θ(k)", label = "")
    title!("Parte inicial: primeros 100 pasos")
    hline!([θ], lw = 2, color = :red, label = "valor teórico de θ = $θ")
end
begin
    plot(T[(end-99):end], lw = 1, xlabel = L"k", ylabel = L"θ(k)", label = "")
    title!("Parte final: últimos 100 pasos")
    hline!([θ], lw = 2, color = :red, label = "valor teórico de θ = $θ")
end
