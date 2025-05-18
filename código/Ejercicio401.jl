### Ejercicio de Modelos Jerárquicos
### Autor: Dr. Arturo Erdely
### Versión: 2025-05-15


## Cargar paquetes y código externo necesario:

begin
    using Distributions, Plots, LaTeXStrings, Random 
    include("02probestim.jl")
    include("zExponencial.jl") # en misma carpeta
end;



## Simular muestra aleatoria

function simulador(β, θ, n)
    # Simula vector de observaciones de X tales que: 
    # Λ ~ Exponencial(β) con β>0 parámetro de precisión
    # Y | Λ = λ ~ Poisson(λ), λ > 0
    # X | Y = y ~ Binomial(y+1, θ), 0 < θ < 1
    Λ = vaExponencial(0, β)
    simΛ = Λ.sim(n)
    X = zeros(Int, n)
    for i ∈ 1:n 
        Y = rand(Poisson(simΛ[i]), 1)[1]
        X[i] = rand(Binomial(Y+1, θ))
    end
    return X 
end

begin # simular muestra específica
    n = 1_000
    β = 3.5
    θ = 0.8
    Random.seed!(8955) # semilla aleatoria para reproducibilidad de este ejemplo
    xobs = simulador(β, θ, n)
end

begin # función de masa de probabilidades empírica de X
    X = masaprob(xobs)
    xmin, xmax = extrema(xobs)
    xrange = collect(xmin:xmax)
    xprob = X.fmp.(xrange)
    bar(xrange, xprob, xlabel = L"x", ylabel = L"\mathbb{P}(X=x)", legend = false)
end


function ev(β, θ) 
    # valores teóricos de E(X) y V(X)
    E = θ*(1 + 1/β)
    V = E*(1 - θ + θ/β)
    return(E,V)
end

ev(β, θ) # valores teóricos de E(X) y V(X)
mean(xobs), var(xobs) # estimaciones muestrales de E(X) y V(X)

function momβθ(μ,σ²) 
    # valor de los párametros dados E(X) y V(X)
    c = (1 - σ²/μ) / μ
    β = (1 + c) / (1 - c)
    θ = β*μ / (1 + β)
    return (β, θ)
end

β, θ # valores teóricos de los parámetros
momβθ(mean(xobs), var(xobs)) # estimaciones por método de momentos



## Método ABC 

function ABC(muestra::Vector{Int64}, nsim = 1_000_000, nselec = 100)
    nobs = length(muestra)
    δ(u, v) = √sum((u - v) .^2) # distancia euclidiana
    dist = zeros(nsim) # inicializar vector de distancias
    βmom, θmom = momβθ(mean(muestra), var(muestra))
    priorβ = rand(Uniform(0, max(1.0, 2*βmom)), nsim)
    if θmom ≤ 0.5 
        priorθ = rand(Uniform(0, 2*θmom), nsim)
    else
        priorθ = rand(Uniform(2*θmom - 1, 1), nsim)
    end
    for i ∈ 1:nsim 
        xsim = simulador(priorβ[i], priorθ[i], nobs)
        βmomsim, θmomsim = momβθ(mean(xsim), var(xsim))
        dist[i] = δ([βmomsim, θmomsim], [βmom, θmom])
    end
    iSelec = sortperm(dist)[1:nselec] # simulaciones a seleccionar
    postβ = priorβ[iSelec]
    postθ = priorθ[iSelec]
    return (β = postβ, θ = postθ)
end

@time post = ABC(xobs, 100_000); # nsim=100_000 ≈ 30 segundos
βpost = median(post.β) # mediana a posteriori de β
θpost = median(post.θ) # medana a posteriori de θ
begin
    # densidad a posteriori de β
    histogram(post.β, color = :yellow, label = "densidad a posteriori")
    xaxis!(L"β")
    vline!([β], lw = 4, color = :blue, label = "valor teórico")
    vline!([βpost], lw = 4, color = :red, label = "mediana a posteriori")
end
begin
    # densidad a posteriori de θ
    histogram(post.θ, color = :yellow, label = "densidad a posteriori")
    xaxis!(L"θ")
    vline!([θ], lw = 4, color = :blue, label = "valor teórico")
    vline!([θpost], lw = 4, color = :red, label = "mediana a posteriori")
end


## Experimentos

begin # simular muestra específica (ya sin semilla aleatoria)
    n = 1_000
    β = 0.1
    θ = 0.2
    xobs = simulador(β, θ, n)
end

begin # función de masa de probabilidades empírica de X
    X = masaprob(xobs)
    xmin, xmax = extrema(xobs)
    xrange = collect(xmin:xmax)
    xprob = X.fmp.(xrange)
    bar(xrange, xprob, xlabel = L"x", ylabel = L"\mathbb{P}(X=x)", legend = false)
end

momβθ(mean(xobs), var(xobs)) # estimaciones por método de momentos

@time post = ABC(xobs, 100_000); # nsim=100_000 ≈ 30 segundos
βpost = median(post.β) # mediana a posteriori de β
θpost = median(post.θ) # medana a posteriori de θ
begin
    # densidad a posteriori de β
    histogram(post.β, color = :yellow, label = "densidad a posteriori")
    xaxis!(L"β")
    vline!([β], lw = 4, color = :blue, label = "valor teórico")
    vline!([βpost], lw = 4, color = :red, label = "mediana a posteriori")
end
begin
    # densidad a posteriori de θ
    histogram(post.θ, color = :yellow, label = "densidad a posteriori")
    xaxis!(L"θ")
    vline!([θ], lw = 4, color = :blue, label = "valor teórico")
    vline!([θpost], lw = 4, color = :red, label = "mediana a posteriori")
end

