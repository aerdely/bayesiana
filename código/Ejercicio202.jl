### Ejemplo muestreador de Gibbs
### Autor: Dr. Arturo Erdely
### Versión: 2025-04-21

#=
    Binomial(r,θ)  
    r ∈ {1,2,…}
    0 < θ < 1
=#


## Paquetes y código necesarios

using Distributions, Random, Plots, LaTeXStrings, QuadGK
include("02probestim.jl")
include("03discreta.jl")


## Simular muestra

r, θ = 7, 0.2; # valor teórico de los parámetros desconocidos
X = Binomial(r,θ);
n = 100; # tamaño de muestra a simular
xx = rand(X, n); # simular muestra observada
println(xx)
sx = sum(xx)
xmax = maximum(xx)


## A posteriori para r condicional en θ

function logLπ(r,θ)
    α = 2.0
    if r ≥ xmax 
        ss = 0.0 
        for i ∈ 1:n 
            for k ∈ 0:(xx[i]-1)
                ss += log(r-k) - log(xx[i]-k)
            end
        end
        return sx*log(θ) + (n*r - sx)*log(1-θ) + log(1/(r^α) - 1/((r+1)^α)) + ss
    else
        return 0.0
    end
end

logLπ(7, 0.2)
logLπ.(collect(0:10), 0.2)

function pr_θ(r,θ)
    g(z) = exp(logLπ(floor(z), θ))
    C = quadgk(g, xmax, Inf)[1]
    return g(r) / C
end

pr_θ(7, 0.20)
[collect(xmax:10) pr_θ.(collect(xmax:10), 0.2)]
sum(pr_θ.(collect(xmax:10), 0.2))

function simr_θ(nsim, θ)
    fmp(r) = pr_θ(r + xmax - 1, θ)
    return simDiscreta(fmp, nsim) .+ xmax .- 1
end

@time simr_θ(1, 0.2) 

println(simr_θ(20, 0.2))





## Muestreador de Gibbs 

@time begin # generar cadenas de Markov
    rG = xmax 
    nsim = 100
    T = zeros(nsim)
    R = zeros(Int, nsim)
    for k ∈ 1:nsim
        T[k] = rand(Beta(1/2 + sx, 1/2 + n*rG - sx), 1)[1]
        R[k] = simr_θ(1, T[k])[1]
        rG = R[k]
    end    
end # 100 sim ≈ 14 segundos, 1000 sim ≈ 578 segundos ≈ 10 minutos

begin # graficando las cadenas de Markov marginales
    plot(R, lw = 1, legend = false, ylabel = L"r", color = :blue, yticks = 0:1:maximum(R))
    hline!([0], color = :lightgray, lw = 0.1)
    p1 = hline!([r], color = :red)
    plot(T, lw = 1, legend = false, ylabel = L"θ", color = :darkgreen)
    p2 = hline!([θ], color = :red)
    plot(p1, p2, layout = (2,1))
end
# savefig("Ejercicio202A.pdf")

@show((r,θ));
mean(R[350:end]), mean(T[350:end])
median(R[350:end]), median(T[350:end])


begin # graficando la cadena de Markov bivariada, primeros 100 pasos
    plot(R[1:100],T[1:100], label = "")
    scatter!(R[1:100],T[1:100], ms = 1.5, label = "")
    xaxis!(L"r"); yaxis!(L"θ")
    scatter!([R[1]], [T[1]], label = "inicio (r₁,θ₁)", color = :yellow)
end
# savefig("Ejercicio202B.pdf")

begin # graficando toda la cadena de Markov bivariada 
    plot(R, T, label = "")
    scatter!(R, T, ms = 1, label = "")
    xaxis!(L"r"); yaxis!(L"θ")
    scatter!([R[1]], [T[1]], label = "inicio (r₁,θ₁)", color = :yellow)
end
# savefig("Ejercicio202C.pdf")


begin # fmp a posteriori para r
    dR = masaprob(R[350:end])
    rmin, rmax = extrema(dR.valores)
    rval = collect(rmin:rmax)
    bar(rval, dR.fmp.(rval), label = "probabilidad a posteriori", color = :cyan)
    xaxis!(L"r"); yaxis!(L"p(r\,|\,\mathbf{x}_{obs})")
    scatter!([median(R[350:end])], [0.003], ms = 5, mc = :blue, label = "mediana a posteriori = $(median(R[350:end]))")
    vline!([r], color = :red, lw = 2, label = "valor teórico r = $r")
end
# savefig("Ejercicio202D.pdf")

begin # fdp a posteriori para θ
    dT = densprob(T[350:end])
    θmin, θmax = dT.min, dT.max
    θval = collect(range(θmin, θmax, length = 1_000))
    plot(θval, dT.fdp.(θval), lw = 2, color = :green, label = "densidad a posteriori")
    xaxis!(L"θ"); yaxis!(L"p(θ\,|\,\mathbf{x}_{obs})")
    scatter!([median(T[350:end])], [0.003], ms = 5, mc = :blue, label = "mediana a posteriori = $(round(median(T[350:end]), digits = 4))")
    vline!([θ], color = :red, lw = 2, label = "valor teórico θ = $θ")
end
# savefig("Ejercicio202E.pdf")
