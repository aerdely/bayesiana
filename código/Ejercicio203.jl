### Ejemplo muestreador de Gibbs
### Autor: Dr. Arturo Erdely
### Versión: 2025-04-17

#=
    Regresión lineal simple bajo normalidad

    Y|X=x ≡ Yₓ := α + βx  + ε 

    ε ~ Normal(0, λ),  V(ε) = 1/λ
    α, β ∈ ℜ
=#


## Paquetes y código necesarios

using Distributions, Random, Plots, LaTeXStrings
include("02probestim.jl")



## Simular muestra

begin
    α, β, λ = 2.0, -5.0, 0.01 # valor teórico de los parámetros desconocidos
    ε = Normal(0, 1/√λ)
    n = 300; # tamaño de muestra a simular
    xx = collect(range(-5.0, 7.0, length = n))
    ℓ(x) = α + β*x
    # simular muestra observada
    ee = rand(ε, n)
    yy = α .+ β.*xx .+ ee
    scatter(xx, yy, ms = 3, label = "(x, Yₓ)")
    xaxis!(L"x"); yaxis!(L"Y_{\!\!x}")
    plot!(xx, ℓ.(xx), label = "α + βx")
end

begin # estadísticos
    sx = sum(xx)
    sx2 = sum(xx .^ 2)
    sxy = sum(xx .* yy)
    my = mean(yy)
    mx = mean(xx)
    sdc(a, b) = sum((yy .- (a .+ b.*xx)) .^ 2)
    # estimaciones de α y β por mínimos cuadrados ordinarios
    bmc = sum((yy .- my) .* (xx .- mx)) / sum((xx .- mx).^2)
    amc = my - bmc*mx
    # estimación de λ
    lmc = (n-2) / sum((yy .- amc .- bmc.*xx).^2)
    amc, bmc, lmc
end



## Muestreador de Gibbs 

@time begin # generar cadenas de Markov
    nsim = 10_000 
    A = zeros(nsim)
    B = zeros(nsim)
    L = zeros(nsim)
    αG, βG = 0.0, 0.0 # valores iniciales
    for k ∈ 1:nsim 
        L[k] = rand(Gamma(n/2 + 1, 1/(sdc(αG, βG)/2)), 1)[1]
        A[k] = rand(Normal(my - βG, n*L[k]), 1)[1]
        B[k] = rand(Normal((sxy - A[k]*sx)/sx2, L[k]/sx2), 1)[1]
        aG, βG = A[k], B[k] 
    end
end

begin # Muestreador de Gibbs para α
    plot(A, label = "cadena de Gibbs", ylabel = "α")
    hline!([α], label = "valor teórico", color = :red)
    hline!([amc], label = "estimación MCO", color = :violet)
end
begin # Muestreador de Gibbs para α (parte inicial)
    plot(A[1:1_000], label = "cadena de Gibbs", ylabel = "α")
    hline!([α], label = "valor teórico", color = :red)
    hline!([amc], label = "estimación MCO", color = :violet)
end

begin # Muestreador de Gibbs para β
    plot(B, label = "cadena de Gibbs", ylabel = "β")
    hline!([β], label = "valor teórico", color = :red)
    hline!([bmc], label = "estimación MCO", color = :violet)
end
begin # Muestreador de Gibbs para β (parte inicial)
    plot(B[1:1_000], label = "cadena de Gibbs", ylabel = "β")
    hline!([β], label = "valor teórico", color = :red)
    hline!([bmc], label = "estimación MCO", color = :violet)
end

begin # Muestreador de Gibbs para λ
    plot(L, label = "cadena de Gibbs", ylabel = "λ")
    hline!([λ], label = "valor teórico", color = :red)
    hline!([lmc], label = "estimación MCO", color = :violet)
end
begin # Muestreador de Gibbs para λ (parte inicial)
    plot(L[1:1_000], label = "cadena de Gibbs", ylabel = "λ")
    hline!([λ], label = "valor teórico", color = :red)
    hline!([lmc], label = "estimación MCO", color = :violet)
end


## Distribuciones a posteriori 

begin # fdp a posteriori para α
    dA = densprob(A[1_000:end])
    αval = collect(range(dA.min, dA.max, length = 1_000))
    plot(αval, dA.fdp.(αval), lw = 2, color = :green, label = "densidad a posteriori", legend = :topright)
    xaxis!(L"α"); yaxis!(L"p(α\,|\,\mathbf{y}_{obs})")
    scatter!([median(A[1_000:end])], [0.003], ms = 5, mc = :blue, label = "mediana a posteriori = $(round(median(A[1_000:end]), digits = 4))")
    vline!([α], color = :red, lw = 2, label = "valor teórico α = $α")
    vline!([amc], color = :violet, lw = 2, label = "estimación MCO = $(round(amc, digits = 4))")
end

begin # fdp a posteriori para β
    dB = densprob(B[1_000:end])
    βval = collect(range(dB.min, dB.max, length = 1_000))
    plot(βval, dB.fdp.(βval), lw = 2, color = :green, label = "densidad a posteriori", legend = :topleft)
    xaxis!(L"β"); yaxis!(L"p(β\,|\,\mathbf{y}_{obs})")
    scatter!([median(B[1_000:end])], [0.003], ms = 5, mc = :blue, label = "mediana a posteriori = $(round(median(B[1_000:end]), digits = 4))")
    vline!([β], color = :red, lw = 2, label = "valor teórico β = $β")
    vline!([bmc], color = :violet, lw = 2, label = "estimación MCO = $(round(bmc, digits = 4))")
end

begin # fdp a posteriori para λ
    dL = densprob(L[1_000:end])
    λval = collect(range(dL.min, dL.max, length = 1_000))
    plot(λval, dL.fdp.(λval), lw = 2, color = :green, label = "densidad a posteriori", legend = :topright)
    xaxis!(L"λ"); yaxis!(L"p(λ\,|\,\mathbf{y}_{obs})")
    scatter!([median(L[1_000:end])], [0.003], ms = 5, mc = :blue, label = "mediana a posteriori = $(round(median(L[1_000:end]), digits = 4))")
    vline!([λ], color = :red, lw = 2, label = "valor teórico λ = $λ")
    vline!([lmc], color = :violet, lw = 2, label = "estimación MCO = $(round(lmc, digits = 4))")
end

