# Ejercicio 1.6 e)

#=
    Exponencial(μ,θ)    μ ∈ ℜ    θ > 0
=#

include("xbExponencial.jl")
using QuadGK, Plots, LaTeXStrings


### Caso: θ > 0, μ = 0

@doc bExpoEsc # consultar documentación 

θ = 3.7; # fijando valor teórico para experimentar

# se usa 1/θ por la reparametrización 
# utilizada en el paquete `Distributions`
X = Exponential(1/θ); 
@doc Exponential 

n = 100; # tamaño de muestra a simular
xx = rand(X, n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori

## modelos bayesianos:

prior = bExpoEsc(α = αpriori, β = βpriori);
post = bExpoEsc(xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpoEsc(xobs = xx);

keys(post)

prior.familia
prior.conjugada
post.conjugada
postNoinfo.conjugada 

prior.α
prior.β
prior.n
prior.muestra

post.α
post.β
post.n
post.muestra

postNoinfo.α
postNoinfo.β
postNoinfo.n
postNoinfo.muestra

## densidades para θ

begin
    t = range(0.0, prior.q(0.99), length = 1_000)
    plot(t, prior.d.(t), lw = 3, label = "a priori", color = :gray)
    xaxis!(L"\theta")
    yaxis!("densidad")
    plot!(t, post.d.(t), lw = 3, label = "a posteriori", color = :red)
    plot!(t, postNoinfo.d.(t), lw = 1, label = "a posteriori no info", color = :green)
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

begin
    t = range(post.q(0.001), post.q(0.999), length = 1_000)
    plot(t, post.d.(t), lw = 3, label = "a priori", color = :red)
    xaxis!(L"\theta")
    yaxis!("densidad")
    plot!(t, postNoinfo.d.(t), lw = 2, label = "a posteriori no info", color = :green)
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

## estimación puntual de θ vía la mediana:

θ # valor teórico, como referencia
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 


## funciones de distribución para θ

begin
    t = range(0.0, prior.q(0.99), length = 1_000)
    plot(t, prior.p.(t), lw = 3, label = "a priori", color = :gray)
    xaxis!(L"\theta")
    yaxis!("distribución")
    plot!(t, post.p.(t), lw = 3, label = "a posteriori", color = :red)
    plot!(t, postNoinfo.p.(t), lw = 1, label = "a posteriori no info", color = :green)
    pm = prior.q(0.5)
    plot!([0.0,pm], [0.5,0.5], color = :darkgray, label = "")
    plot!([pm,pm], [0.0,0.5], color = :darkgray, label = "")
    scatter!([pm], [0.0], ms = 5, color = :gray, label = "θ* = $(round(pm,digits=2))")
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

begin
    t = range(post.q(0.001), post.q(0.999), length = 1_000)
    plot(t, post.p.(t), lw = 3, label = "a posteriori", color = :red)
    xaxis!(L"\theta")
    yaxis!("distribución")
    plot!(t, postNoinfo.p.(t), lw = 2, label = "a posteriori no info", color = :green)
    p0 = post.q(0.001)
    pm = post.q(0.5)
    plot!([p0,pm], [0.5,0.5], color = :red, label = "")
    plot!([pm,pm], [0.0,0.5], color = :red, label = "")
    scatter!([pm], [0.0], ms = 5, color = :red, label = "θ* = $(round(pm,digits=2))")
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

## simulaciones de θ

begin # a priori
    nsim = 10_000
    θsim = prior.r(nsim)
    θmin, θmax = extrema(θsim)
    θval = range(θmin, θmax, length = 1_000)
    histogram(θsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"θ"); yaxis!("densidad")
    plot!(θval, prior.d.(θval), lw = 3, label = "a priori", color = :darkgray)
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

begin # a posteriori
    nsim = 10_000
    θsim = post.r(nsim)
    θmin, θmax = extrema(θsim)
    θval = range(θmin, θmax, length = 1_000)
    histogram(θsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"θ"); yaxis!("densidad")
    plot!(θval, post.d.(θval), lw = 3, label = "a posteriori", color = :red)
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

begin # a posteriori no informativa
    nsim = 10_000
    θsim = postNoinfo.r(nsim)
    θmin, θmax = extrema(θsim)
    θval = range(θmin, θmax, length = 1_000)
    histogram(θsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"θ"); yaxis!("densidad")
    plot!(θval, post.d.(θval), lw = 3, label = "a posteriori no info", color = :green)
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end


## densidades predictivas para X

# comprobando que integran a 1
quadgk(prior.dp, 0, Inf)
quadgk(post.dp, 0, Inf)
quadgk(postNoinfo.dp, 0, Inf)

begin
    x = range(0.001, quantile(X, 0.995), length = 1_000)
    plot(x, pdf(X, x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(x, prior.dp.(x), lw = 2, color = :gray, label = "a priori")
    plot!(x, post.dp.(x), lw = 2, color = :red, label = "a posteriori")
end

## estimación puntual predictiva para X vía la mediana: 
median(X) # valor teórico, como referencia
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa


## distribuciones predictivas para X 

begin
    x = range(0.001, quantile(X, 0.995), length = 1_000)
    plot(x, cdf(X, x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x"); yaxis!("distribución predictiva")
    plot!(x, prior.pp.(x), lw = 2, color = :gray, label = "a priori")
    plot!(x, post.pp.(x), lw = 2, color = :red, label = "a posteriori")
end


## simulaciones predictivas para X 

begin # a priori
    nsim = 1_000
    xsim = prior.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad")
    plot!(xval, prior.dp.(xval), lw = 3, label = "a priori", color = :darkgray)
end

begin # a posteriori
    nsim = 10_000
    xsim = post.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori", color = :red)
end

begin # a posteriori no info
    nsim = 10_000
    xsim = postNoinfo.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori no info", color = :green)
end



## Caso: μ ∈ ℜ, θ = 1

# … desarróllalo


## Caso: μ ∈ ℜ, θ > 0

# … desarróllalo
