# Ejercicio 1.6 e)

#=
    Exponencial(μ,θ)    μ ∈ ℜ    θ > 0

    Línea | Caso
    ------|-------------------
      19  | μ = 0, θ > 0 desconocido
     221  | μ ∈ ℜ desconocido, θ = 1
     438  | μ ∈ ℜ conocido, θ > 0 desconocido
     620  | μ ∈ ℜ desconocido, θ > 0 conocido          
     822  | μ ∈ ℜ, θ > 0, ambos parámetros desconocidos
=#

using QuadGK, HCubature, Plots, LaTeXStrings
include("xbExponencial.jl")


### Caso: μ = 0, θ > 0 desconocido

@doc bExpoEsc # consultar documentación 

@doc vaExponencial # consultar documentación

θ = 3.7 # fijando valor teórico para experimentar

X = vaExponencial(0.0, θ);
n = 100; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori

## modelos bayesianos:

prior = bExpoEsc(α = αpriori, β = βpriori);
post = bExpoEsc(xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpoEsc(xobs = xx);

keys(post)

prior.familia

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
    plot(t, post.d.(t), lw = 3, label = "a posteriori", color = :red)
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
quadgk(prior.dp, -Inf, Inf)
quadgk(post.dp, -Inf, Inf)
quadgk(postNoinfo.dp, -Inf, Inf)

begin
    x = range(0.001, X.ctl(0.995), length = 1_000)
    plot(x, X.fdp.(x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(x, prior.dp.(x), lw = 2, color = :gray, label = "a priori")
    plot!(x, post.dp.(x), lw = 2, color = :red, label = "a posteriori")
end

## estimación puntual predictiva para X vía la mediana: 

X.mediana # valor teórico, como referencia
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa


## distribuciones predictivas para X 

begin
    x = range(0.001, X.ctl(0.995), length = 1_000)
    plot(x, X.fda.(x), lw = 3, color = :blue, label = "teórica")
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
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, prior.dp.(xval), lw = 3, label = "a priori", color = :darkgray)
end

begin # a posteriori
    nsim = 10_000
    xsim = post.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori", color = :red)
end

begin # a posteriori no info
    nsim = 10_000
    xsim = postNoinfo.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori no info", color = :green)
end




### Caso: μ ∈ ℜ desconocido, θ = 1

@doc bExpoLoc # consultar documentación 

μ = -2.1 # fijando valor teórico para experimentar

X = vaExponencial(μ, 1.0);
n = 10; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 10.0, 0.06 # hiperparámetros a priori

## modelos bayesianos:

prior = bExpoLoc(α = αpriori, β = βpriori);
post = bExpoLoc(xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpoLoc(xobs = xx);
prior.familia
keys(post)

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


## algunas verificaciones 

# comprobando que densidades integran a 1
quadgk(prior.d, -Inf, Inf)
quadgk(post.d, -Inf, Inf)
quadgk(postNoinfo.d, -Inf, Inf)
quadgk(prior.dp, -Inf, Inf)
quadgk(post.dp, -Inf, Inf)
quadgk(postNoinfo.dp, -Inf, Inf)

# comprobando límites de funciones de distribución y de cuantiles
prior.p(-Inf), prior.p(Inf)
post.p(-Inf), post.p(Inf)
postNoinfo.p(-Inf), postNoinfo.p(Inf)
prior.pp(-Inf), prior.pp(Inf)
post.pp(-Inf), post.pp(Inf)
postNoinfo.pp(-Inf), postNoinfo.pp(Inf)
prior.q(0), prior.q(1)
post.q(0), post.q(1)
postNoinfo.q(0), postNoinfo.q(1)
prior.qp(0), prior.qp(1)
post.qp(0), post.qp(1)
postNoinfo.qp(0), postNoinfo.qp(1)


## estimación puntual de μ vía la mediana:

μ # valor teórico, como referencia
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 


## densidades para μ

begin
    t = range(prior.q(0.01), prior.q(0.99), length = 1_000)
    plot(t, prior.d.(t), lw = 3, label = "a priori", color = :gray)
    xaxis!(L"μ")
    yaxis!("densidad")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin
    t = range(post.q(0.001), post.q(0.999), length = 1_000)
    plot(t, post.d.(t), lw = 3, label = "a posteriori", color = :red)
    xaxis!(L"μ")
    yaxis!("densidad")
    plot!(t, postNoinfo.d.(t), lw = 2, label = "a posteriori no info", color = :green)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## funciones de distribución para μ

begin
    t = range(prior.q(0.01), prior.q(0.99), length = 1_000)
    plot(t, prior.p.(t), lw = 3, label = "a priori", color = :gray)
    xaxis!(L"μ")
    yaxis!("distribución")
    pm = prior.q(0.5)
    plot!([prior.q(0.01),pm], [0.5,0.5], color = :darkgray, label = "")
    plot!([pm,pm], [0.0,0.5], color = :darkgray, label = "")
    scatter!([pm], [0.0], ms = 5, color = :gray, label = "μ* = $(round(pm,digits=2))")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin
    t = range(post.q(0.001), post.q(0.999), length = 1_000)
    plot(t, post.p.(t), lw = 3, label = "a posteriori", color = :red)
    xaxis!(L"\mu")
    yaxis!("distribución")
    plot!(t, postNoinfo.p.(t), lw = 2, label = "a posteriori no info", color = :green)
    p0 = post.q(0.001)
    pm = post.q(0.5)
    plot!([p0,pm], [0.5,0.5], color = :red, label = "")
    plot!([pm,pm], [0.0,0.5], color = :red, label = "")
    scatter!([pm], [0.0], ms = 5, color = :red, label = "μ* = $(round(pm,digits=2))")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## simulaciones de μ

begin # a priori
    nsim = 10_000
    μsim = prior.r(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, prior.d.(μval), lw = 3, label = "a priori", color = :darkgray)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin # a posteriori
    nsim = 10_000
    μsim = post.r(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, post.d.(μval), lw = 3, label = "a posteriori", color = :red)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin # a posteriori no informativa
    nsim = 10_000
    μsim = postNoinfo.r(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, post.d.(μval), lw = 3, label = "a posteriori no info", color = :green)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## estimación puntual predictiva vía la mediana: 

X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa


## densidades predictivas para X

begin
    x = range(X.ctl(0.0001), X.ctl(0.995), length = 1_000)
    plot(x, X.fdp.(x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(x, prior.dp.(x), lw = 2, color = :gray, label = "a priori")
    plot!(x, post.dp.(x), lw = 2, color = :red, label = "a posteriori")
end


## distribuciones predictivas para X 

begin
    x = range(X.ctl(0.005), X.ctl(0.995), length = 1_000)
    plot(x, X.fda.(x), lw = 3, color = :blue, label = "teórica")
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
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, prior.dp.(xval), lw = 3, label = "a priori", color = :darkgray)
end

begin # a posteriori
    nsim = 10_000
    xsim = post.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori", color = :red)
end

begin # a posteriori no info
    nsim = 10_000
    xsim = postNoinfo.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori no info", color = :green)
end



### Caso: μ ∈ ℜ conocido, θ > 0 desconocido

@doc bExpo1 # consultar documentación 

μ = -1.5 # valor del parámetro conocido
θ = 3.7 # fijando valor teórico para experimentar

X = vaExponencial(μ,θ);
n = 300; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori

## modelos bayesianos:

prior = bExpo1(μ, α = αpriori, β = βpriori);
post = bExpo1(μ, xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpo1(μ, xobs = xx);
prior.familia
keys(post)

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
    plot(t, post.d.(t), lw = 3, label = "a posteriori", color = :red)
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
quadgk(prior.dp, -Inf, Inf)
quadgk(post.dp, μ, Inf)
quadgk(postNoinfo.dp, μ, Inf)

begin
    x = range(X.ctl(0.001), X.ctl(0.995), length = 1_000)
    plot(x, X.fdp.(x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(x, prior.dp.(x), lw = 2, color = :gray, label = "a priori")
    plot!(x, post.dp.(x), lw = 2, color = :red, label = "a posteriori")
end

## estimación puntual predictiva para X vía la mediana: 

X.mediana # valor teórico, como referencia
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa


## distribuciones predictivas para X 

begin
    x = range(X.ctl(0.001), X.ctl(0.995), length = 1_000)
    plot(x, X.fda.(x), lw = 3, color = :blue, label = "teórica")
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
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, prior.dp.(xval), lw = 3, label = "a priori", color = :darkgray)
end

begin # a posteriori
    nsim = 10_000
    xsim = post.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori", color = :red)
end

begin # a posteriori no info
    nsim = 10_000
    xsim = postNoinfo.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori no info", color = :green)
end



### Caso: μ ∈ ℜ desconocido, θ > 0 conocido

@doc bExpo2 # consultar documentación 
 
μ = -1.5 # fijando valor teórico para experimentar
θ = 3.7 # valor del parámetro conocido

X = vaExponencial(μ,θ);
n = 20; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 10*θ, 0.06/θ # hiperparámetros a priori

## modelos bayesianos:

prior = bExpo2(θ; α = αpriori, β = βpriori);
post = bExpo2(θ; xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpo2(θ; xobs = xx);
prior.familia
keys(post)

## algunas verificaciones 

# comprobando que densidades integran a 1
quadgk(prior.d, -Inf, Inf)
quadgk(post.d, -Inf, Inf)
quadgk(postNoinfo.d, -Inf, Inf)
quadgk(prior.dp, -Inf, Inf)
quadgk(post.dp, -Inf, Inf)
quadgk(postNoinfo.dp, -Inf, Inf)

# comprobando límites de funciones de distribución y de cuantiles
prior.p(-Inf), prior.p(Inf)
post.p(-Inf), post.p(Inf)
postNoinfo.p(-Inf), postNoinfo.p(Inf)
prior.pp(-Inf), prior.pp(Inf)
post.pp(-Inf), post.pp(Inf)
postNoinfo.pp(-Inf), postNoinfo.pp(Inf)
prior.q(0), prior.q(1)
post.q(0), post.q(1)
postNoinfo.q(0), postNoinfo.q(1)
prior.qp(0), prior.qp(1)
post.qp(0), post.qp(1)
postNoinfo.qp(0), postNoinfo.qp(1)


## estimación puntual de μ vía la mediana:

μ # valor teórico, como referencia
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 


## densidades para μ

begin
    t = range(prior.q(0.01), prior.q(0.99), length = 1_000)
    plot(t, prior.d.(t), lw = 3, label = "a priori", color = :gray)
    xaxis!(L"μ")
    yaxis!("densidad")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin
    t = range(post.q(0.001), post.q(0.999), length = 1_000)
    plot(t, post.d.(t), lw = 3, label = "a posteriori", color = :red)
    xaxis!(L"μ")
    yaxis!("densidad")
    plot!(t, postNoinfo.d.(t), lw = 2, label = "a posteriori no info", color = :green)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## funciones de distribución para μ

begin
    t = range(prior.q(0.01), prior.q(0.99), length = 1_000)
    plot(t, prior.p.(t), lw = 3, label = "a priori", color = :gray)
    xaxis!(L"μ")
    yaxis!("distribución")
    pm = prior.q(0.5)
    plot!([prior.q(0.01),pm], [0.5,0.5], color = :darkgray, label = "")
    plot!([pm,pm], [0.0,0.5], color = :darkgray, label = "")
    scatter!([pm], [0.0], ms = 5, color = :gray, label = "μ* = $(round(pm,digits=2))")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin
    t = range(post.q(0.001), post.q(0.999), length = 1_000)
    plot(t, post.p.(t), lw = 3, label = "a posteriori", color = :red)
    xaxis!(L"\mu")
    yaxis!("distribución")
    plot!(t, postNoinfo.p.(t), lw = 2, label = "a posteriori no info", color = :green)
    p0 = post.q(0.001)
    pm = post.q(0.5)
    plot!([p0,pm], [0.5,0.5], color = :red, label = "")
    plot!([pm,pm], [0.0,0.5], color = :red, label = "")
    scatter!([pm], [0.0], ms = 5, color = :red, label = "μ* = $(round(pm,digits=2))")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## simulaciones de μ

begin # a priori
    nsim = 10_000
    μsim = prior.r(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, prior.d.(μval), lw = 3, label = "a priori", color = :darkgray)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin # a posteriori
    nsim = 10_000
    μsim = post.r(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, post.d.(μval), lw = 3, label = "a posteriori", color = :red)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

begin # a posteriori no informativa
    nsim = 10_000
    μsim = postNoinfo.r(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, post.d.(μval), lw = 3, label = "a posteriori no info", color = :green)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## estimación puntual predictiva vía la mediana: 

X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa


## densidades predictivas para X

begin
    x = range(X.ctl(0.005), X.ctl(0.995), length = 1_000)
    plot(x, X.fdp.(x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(x, prior.dp.(x), lw = 2, color = :gray, label = "a priori")
    plot!(x, post.dp.(x), lw = 2, color = :red, label = "a posteriori")
end


## distribuciones predictivas para X 

begin
    x = range(X.ctl(0.005), X.ctl(0.995), length = 1_000)
    plot(x, X.fda.(x), lw = 3, color = :blue, label = "teórica")
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
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, prior.dp.(xval), lw = 3, label = "a priori", color = :darkgray)
end

begin # a posteriori
    nsim = 10_000
    xsim = post.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori", color = :red)
end

begin # a posteriori no info
    nsim = 10_000
    xsim = postNoinfo.rp(nsim)
    xmin, xmax = extrema(xsim)
    xval = range(xmin, xmax, length = 1_000)
    histogram(xsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"x"); yaxis!("densidad predictiva")
    plot!(xval, post.dp.(xval), lw = 3, label = "a posteriori no info", color = :green)
end



### Caso: μ ∈ ℜ, θ > 0, ambos parámetros desconocidos

@doc bExpo # consultar documentación 

μ,θ = -1.5, 3.7; # valor teórico de los parámetros desconocidos
X = vaExponencial(μ,θ);
n = 100; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada

## modelo bayesiano no informativo:

post = bExpo(xx);
keys(post)
post.familia

## marginal de μ a posteriori no informativa

μ # valor teórico
post.qμ(0.5) # estimación puntual a posteriori no informativa

# densidad de μ
begin
    t = range(post.qμ(0.001), post.qμ(0.999), length = 1_000)
    plot(t, post.dμ.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"μ")
    yaxis!("densidad")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end
quadgk(post.dμ, -Inf, post.xmin)

# función de distribución de μ
begin
    t = range(post.qμ(0.001), post.qμ(0.999), length = 1_000)
    plot(t, post.pμ.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"\mu")
    yaxis!("distribución")
    p0 = post.qμ(0.001)
    pm = post.qμ(0.5)
    plot!([p0,pm], [0.5,0.5], color = :red, label = "")
    plot!([pm,pm], [0.0,0.5], color = :red, label = "")
    scatter!([pm], [0.0], ms = 5, color = :red, label = "μ* = $(round(pm,digits=2))")
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end

# simulaciones de μ
begin 
    nsim = 10_000
    μsim = post.rμ(nsim)
    μmin, μmax = extrema(μsim)
    μval = range(μmin, μmax, length = 1_000)
    histogram(μsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"μ"); yaxis!("densidad")
    plot!(μval, post.dμ.(μval), lw = 3, label = "a posteriori no info", color = :red)
    scatter!([μ], [0], ms = 5, color = :blue, label = "μ = $μ")
end


## marginal de θ a posteriori no informativa

θ # valor teórico
post.qθ(0.5) # estimación puntual a posteriori no informativa

# densidad de θ
begin
    t = range(post.qθ(0.001), post.qθ(0.999), length = 1_000)
    plot(t, post.dθ.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"θ")
    yaxis!("densidad")
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end
quadgk(post.dθ, 0, Inf)

# función de distribución de θ
begin
    t = range(post.qθ(0.001), post.qθ(0.999), length = 1_000)
    plot(t, post.pθ.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"θ")
    yaxis!("distribución")
    p0 = post.qθ(0.001)
    pm = post.qθ(0.5)
    plot!([p0,pm], [0.5,0.5], color = :red, label = "")
    plot!([pm,pm], [0.0,0.5], color = :red, label = "")
    scatter!([pm], [0.0], ms = 5, color = :red, label = "θ* = $(round(pm,digits=2))")
    scatter!([θ], [0], ms = 5, color = :blue, label = "θ = $θ")
end

# simulaciones de θ
begin 
    nsim = 10_000
    θsim = post.rθ(nsim)
    θmin, θmax = extrema(θsim)
    θval = range(θmin, θmax, length = 1_000)
    histogram(θsim, normalize = true, label = "simulada", color = :white)
    xaxis!(L"θ"); yaxis!("densidad")
    plot!(θval, post.dθ.(θval), lw = 3, label = "a posteriori no info", color = :red)
    scatter!([θ], [0.0], ms = 5, color = :blue, label = "θ = $θ")
end


## distribución conjunta a posteriori de (μ,θ)

sim_μθ = post.r(10_000) # simulación de conjunta (μ,θ)

μ # valor teórico
post.qμ(0.5) # estimación puntual vía la mediana
median(sim_μθ[:, 1]) # estimación puntual vía simulación

θ # valor teórico 
post.qθ(0.5) # estimación puntual vía la mediana
median(sim_μθ[:, 2]) # estimación puntual vía simulación

# densidad conjunta de (μ,θ) a posteriori (3D)
begin 
    ngrid = 100
    x = collect(range(quantile(sim_μθ[:, 1], 0.03), maximum(sim_μθ[:, 1]), length = ngrid))
    y = collect(range(minimum(sim_μθ[:, 2]), maximum(sim_μθ[:, 2]), length = ngrid))
    z = zeros(ngrid, ngrid)
    for i ∈ eachindex(x), j ∈ eachindex(y)
        z[j, i] = post.d(x[i], y[j])
    end
    surface(x, y, z, xlabel = L"μ", ylabel = L"θ", size = (500, 500), 
            zlabel = L"p(μ,θ\,|\,\mathbf{x}_{obs})", title = "densidad a posteriori")
    scatter!([μ], [θ], [0.0], ms = 5, mc = :green, label = "(μ,θ) = ($μ,$θ)")
end

# densidad conjunta de (μ,θ) a posteriori (2D)
begin
    contour(x, y, z, xlabel = L"μ", ylabel = L"θ", fill = true, 
            title = "densidad a posteriori", size = (500,500))
    scatter!([μ], [θ], ms = 5, mc = :green, label = "(μ,θ) = ($μ,$θ)")    
end

# simulación conjunta de (μ,θ) a posteriori
begin
    scatter(sim_μθ[:, 1], sim_μθ[:, 2], ms = 1, mc = :gray, size = (500,500), label = "")
    title!("Simulación conjunta a posteriori")
    xaxis!(L"μ")
    yaxis!(L"θ")
    scatter!([μ], [θ], ms = 5, mc = :green, label = "(μ,θ) = ($μ,$θ)")
end

# Verificando ∬p(μ,θ|xobs)dμdθ = 1
@doc hcubature # requiere paquete `HCubature` previamente instalado

begin
    f(x) = post.d(x[1],x[2])
    a = [minimum(sim_μθ[:, 1]), minimum(sim_μθ[:, 2])]
    b = [maximum(sim_μθ[:, 1]), maximum(sim_μθ[:, 2])]
    hcubature(f, a, b)
end


## distribución predictiva a posteriori 

# estimación puntual predictiva de la mediana:
X.mediana # teórica
mediana(μ,θ) = μ - log(1/2)/θ # fórmula de la mediana
μ,θ
mediana(μ,θ) # teórica
post.qp(0.5) # estimación puntual
# estimación vía simulación
begin
    ns = size(sim_μθ)[1]
    msim = zeros(ns);
    for i ∈ 1:ns
        msim[i] = mediana(sim_μθ[i,1], sim_μθ[i,2])
    end
    mean(msim) 
end

# densidad predictiva
quadgk(post.dp, -Inf, Inf) # ∫p(x|xobs)dx = 1
begin
    x = collect(range(post.qp(0.001), post.qp(0.99), length = 1_000))
    plot(x, X.fdp.(x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x")
    yaxis!("densidad")
    plot!(x, post.dp.(x), lw = 1.7, color = :red, label = "predictiva")
end

# función de distribución predictiva 
post.pp(-Inf), post.pp(Inf)
begin
    x = collect(range(post.qp(0.001), post.qp(0.99), length = 1_000))
    plot(x, X.fda.(x), lw = 3, color = :blue, label = "teórica")
    xaxis!(L"x")
    yaxis!("función de distribución")
    plot!(x, post.pp.(x), lw = 1.7, color = :red, label = "predictiva")
end

# simulación predictiva 
begin
    nsim = 10_000
    xsim = post.rp(nsim)
    histogram(xsim, normalize = true, color = :white, label = "simulada")
    xaxis!(L"x"); yaxis!("densidad")
    x = collect(range(post.qp(0.001), post.qp(0.99), length = 1_000))
    plot!(x, post.dp.(x), lw = 3, color = :red, label = "predictiva")
end
