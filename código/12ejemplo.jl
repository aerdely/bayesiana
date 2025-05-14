### Ejemplo de modelo jerárquico con ABC
### Autor: Dr. Arturo Erdely
### Versión: 2025-05-10


# Cargar paquetes y código externo necesarios:
begin
    using Random # no requiere instalación previa 
    using Distributions, Plots, LaTeXStrings # previamente instalados
    include("02probestim.jl") # en misma carpeta
end


# Simulamos muestras aleatorias

begin
    nn = [20, 10, 50, 40, 30] # tamaños de muestras
    I = length(nn) # número de grupos
    α = 2 # hiperparámetro
    β = 5 # hiperparámetro
    ϕ = Beta(α, β) # a priori conjugada para Bernoulli
    Random.seed!(444) # semilla aleatoria para reproducibilidad de este ejemplo
    θ = rand(ϕ, I) # parámetros Bernoulli
    xx = fill(Float64[], I)
    for i ∈ 1:I 
        xx[i] = rand(Bernoulli(θ[i]), nn[i])
    end
    xx
end


# Método ABC

function ABC(xx::Vector{Vector{Float64}}, m = 1_000_000, k = 100)
    I = length(xx)
    nn = length.(xx)
    n = sum(nn)
    qq = nn ./ n 
    medias = mean.(xx)
    αsim = rand(Pareto(1, 1), m) .- 1.0
    βsim = rand(Pareto(1, 1), m) .- 1.0
    θsim = zeros(m, I)
    δ = zeros(m)
    δI = zeros(I)
    xxsim = fill(Float64[], I)
    for j ∈ 1:m
        θsim[j, :] = rand(Beta(αsim[j], βsim[j]), I)
        for i ∈ 1:I 
            xxsim[i] = rand(Bernoulli(θsim[j,i]), nn[i])
            δI[i] = abs(mean(xxsim[i]) - medias[i])
        end
        δ[j] = sum(qq .* δI)
    end
    iSelec = sortperm(δ)[1:k]
    return (θ = θsim[iSelec, :], α = αsim[iSelec], β = βsim[iSelec])
end

@time post = ABC(xx, 10_000_000, 100) # m = 10 millones ≈ 15 segundos
post.θ
post.α
post.β


α, β # valores teóricos
αest, βest = median(post.α), median(post.β) # estimaciones puntuales
transpose(θ) # valores teóricos
median(post.θ, dims=1) # estimaciones puntuales
θest = transpose(median(post.θ, dims=1));


# Densidades marginales a posteriori

begin
    g = []
    pα = densprob(post.α)
    vα = collect(range(pα.min, pα.max, length = 1_000))
    plot(vα, pα.fdp.(vα), label = "densidad a posteriori", lw = 3)
    xaxis!(L"α"); yaxis!(L"p(α\,|\,\mathbf{x}_{obs})")
    vline!([α], label = "valor teórico = $α")
    gα = vline!([αest], label = "estimación puntual = $(round(αest, digits = 2))")
    push!(g, gα)
    pβ = densprob(post.β)
    vβ = collect(range(pβ.min, pβ.max, length = 1_000))
    plot(vβ, pβ.fdp.(vβ), label = "densidad a posteriori", lw = 3)
    xaxis!(L"β"); yaxis!(L"p(β\,|\,\mathbf{x}_{obs})")
    vline!([β], label = "valor teórico = $β")
    gβ = vline!([βest], label = "estimación puntual = $(round(βest, digits = 2))")
    push!(g, gβ)
    for i ∈ 1:I 
        pθ = densprob(post.θ[:, i])
        vθ = collect(range(pθ.min, pθ.max, length = 1_000))
        plot(vθ, pθ.fdp.(vθ), label = "densidad a posteriori", lw = 3)
        xaxis!("θ[$i]"); yaxis!(L"p(θ\,|\,\mathbf{x}_{obs})")
        vline!([θ[i]], label = "valor teórico = $(round(θ[i], digits = 4))")
        gθ = vline!([θest[i]], label = "estimación puntual = $(round(θest[i], digits = 4))")
        push!(g, gθ)
    end
end

g[1] # α
g[2] # β
g[3] # θ₁
g[4] # θ₂
g[5] # θ₃
g[6] # θ₄
g[7] # θ₅
