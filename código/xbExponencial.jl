#=
    Inferencia bayesiana para la familia 
    Exponencial(μ,θ)  μ ∈ ℜ  θ > 0
=#

using Distributions
include("zExponencial.jl")


## Caso: θ > 0, μ = 0
"""
    bExpoEsc(; xobs = zeros(0), α = 0, β = 0)

Distribuciones a posteriori y predictiva a posteriori para una distribución
Exponencial(μ=0, θ) donde θ > 0 es parámetro de escala desconocido, con
función de densidad de probabilidades:

```math 
f(x|θ) = θexp(-θx),  x > 0
```
donde:

- `xobs` = vector con la muestra aleatoria observada (sin muestra si no se especifica)
- `α, β` = hiperparámetros de la distribución a priori Gamma(α,β)

> Requiere que el paquete `Distributions` sea previamente instalado.

> Depende del archivo `zExponencial.jl`

Entrega un tupla etiquetada con los siguientes elementos:

1. `familia` = distribución de probabilidad
2. `d` = función de densidad a posteriori para θ
3. `p` = función de distribución a posteriori para θ
4. `q` = función de cuantiles a posteriori para θ
5. `r` = función simuladora a posteriori para θ
6. `dp` = función de densidad predictiva a posteriori
7. `pp` = función de distribución predictiva a posteriori
8. `qp` = función de cuantiles predictiva a posteriori 
9. `rp` = función simuladora predictiva a posteriori 
10. `n` = tamaño de la muestra observada 
11. `sx` = suma de la muestra observada 
12. `muestra` = vector de la muestra observada
13. `α, β` = valores de los hiperparámetros a priori 

# Ejemplo
```
θ = 3.7; # valor teórico del parámetro
X = vaExponencial(0.0, θ);
n = 1_000; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori
## modelos bayesianos:
prior = bExpoEsc(α = αpriori, β = βpriori);
post = bExpoEsc(xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpoEsc(xobs = xx);
prior.familia
keys(post)
## estimación puntual de θ vía la mediana:
θ # valor teórico
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 
## estimación puntual predictiva vía la mediana: 
X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa
```
"""
function bExpoEsc(; xobs = zeros(0), α = 0.0, β = 0.0)
    n = length(xobs)
    ∑xᵢ = sum(xobs)
    post = Gamma(α + n, 1 / (β + ∑xᵢ)) # reparametrización λ = 1/θ
    dpost(θ) = pdf(post, θ)
    ppost(θ) = cdf(post, θ)
    qpost(u) = quantile(post, u)
    rpost(m) = rand(post, m)
    dpred(x) = (x > 0)*(α + n)*(((β + ∑xᵢ)/(x + β + ∑xᵢ))^(α+n))/(x + β + ∑xᵢ)
    ppred(x) = (x > 0)*(1 - ((β + ∑xᵢ)/(x + β + ∑xᵢ))^(α+n)) 
    qpred(u) = (0 ≤ u ≤ 1)*(β + ∑xᵢ)*((1-u)^(-1/(α+n))-1)
    rpred(m) = qpred.(rand(m))
    return (familia = "Exponencial de escala θ > 0", d = dpost, p = ppost,
            q = qpost, r = rpost, dp = dpred, pp = ppred, qp = qpred, rp = rpred,
            n = n, sx = ∑xᵢ, muestra = xobs, α = α, β = β)
end



## Caso: μ ∈ ℜ, θ = 1
"""
    bExpoLoc(; xobs = zeros(0), α = Inf, β = 0)

Distribuciones a posteriori y predictiva a posteriori para una distribución
Exponencial(μ, θ=1) donde μ ∈ ℜ es parámetro de localización desconocido,
con función de densidad de probabilidades:

```math 
f(x|μ) = exp(-(x - μ)),  x > μ
```
donde:

- `xobs` = vector con la muestra aleatoria observada (sin muestra si no se especifica)
- `α, β` = hiperparámetros de la distribución a priori: π(μ|α,β) = βexp(β(μ-α)), μ<α

Entrega un tupla etiquetada con los siguientes elementos:

1. `familia` = distribución de probabilidad
2. `d` = función de densidad a posteriori para μ
3. `p` = función de distribución a posteriori para μ
4. `q` = función de cuantiles a posteriori para μ
5. `r` = función simuladora a posteriori para μ
6. `dp` = función de densidad predictiva a posteriori
7. `pp` = función de distribución predictiva a posteriori
8. `qp` = función de cuantiles predictiva a posteriori 
9. `rp` = función simuladora predictiva a posteriori 
10. `n` = tamaño de la muestra observada 
11. `xmin` = valor mínimo de la muestra observada 
12. `muestra` = vector de la muestra observada
13. `α, β` = valores de los hiperparámetros a priori 

# Ejemplo
```
μ = -2.1;
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
## estimación puntual de μ vía la mediana:
μ # valor teórico
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 
## estimación puntual predictiva vía la mediana: 
X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa
```
"""
function bExpoLoc(; xobs = zeros(0), α = Inf, β = 0.0)
    n = length(xobs)
    xmin = n ≥ 1 ? minimum(xobs) : Inf 
    dprior(μ, α, β) = β*exp(β*(μ - α))*(μ < α)
    pprior(t, α, β) = (t < α)*exp(-β*(α - t)) + 1*(t ≥ α)
    qprior(u, α, β) = (α + log(u)/β)*(0 ≤ u ≤ 1)
    rprior(m, α, β) = qprior.(rand(m), α, β)
    αpost = min(α, xmin)
    βpost = β + n
    dpost(μ) = dprior(μ, αpost, βpost)
    ppost(μ) = pprior(μ, αpost, βpost)
    qpost(u) = qprior(u, αpost, βpost)
    rpost(m) = rprior(m, αpost, βpost)
    dpred(x) = exp(-x + (βpost+1)*min(x, αpost) - βpost*αpost)*βpost/(βpost + 1)
    function ppred(t)
        if t ≤ αpost
            return exp(βpost*(t - αpost))/(βpost + 1)
        else
            return (1 - βpost*exp(αpost - t)/(βpost + 1))
        end
    end
    function qpred(u)
        if 0 ≤ u ≤ 1/(βpost + 1)
            return αpost + (log(βpost + 1) + log(u))/βpost
        elseif 1/(βpost + 1) < u ≤ 1 
            return αpost - log(1 + 1/βpost) - log(1-u)
        else
            return NaN 
        end
    end
    rpred(m) = qpred.(rand(m))
    return (familia = "Exponencial de localización μ ∈ ℜ", d = dpost, p = ppost,
            q = qpost, r = rpost, dp = dpred, pp = ppred, qp = qpred, rp = rpred,
            n = n, xmin = xmin, muestra = xobs, α = α, β = β)
end



## Caso: μ ∈ ℜ conocida, θ > 0 desconocida 
"""
    bExpo1(μ; xobs = zeros(0), α = 0, β = 0)

Distribuciones a posteriori y predictiva a posteriori para una distribución
Exponencial(μ,θ) donde μ ∈ ℜ es conocido, pero θ > 0 es desconocido, con
función de densidad de probabilidades:

```math 
f(x|θ) = θexp(-θ(x - μ)),  x > μ
```

donde:

- `μ` = valor conocido de este parámetro 
- `xobs` = vector con la muestra aleatoria observada (sin muestra, si no se especifica)
- `α, β` = hiperparámetros de la distribución a priori Gamma(α,β)

> Utiliza la función `bExpoEsc`

Entrega un tupla etiquetada con los siguientes elementos:

1. `familia` = distribución de probabilidad
2. `d` = función de densidad a posteriori para θ
3. `p` = función de distribución a posteriori para θ
4. `q` = función de cuantiles a posteriori para θ
5. `r` = función simuladora a posteriori para θ
6. `dp` = función de densidad predictiva a posteriori
7. `pp` = función de distribución predictiva a posteriori
8. `qp` = función de cuantiles predictiva a posteriori 
9. `rp` = función simuladora predictiva a posteriori 
10. `n` = tamaño de la muestra observada 
11. `muestra` = vector de la muestra observada
12. `α, β` = valores de los hiperparámetros a priori 

# Ejemplo
```
μ = -1.5; # valor del parámetro conocido
θ = 3.7; # valor teórico del parámetro desconocido
X = vaExponencial(μ,θ);
n = 1_000; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori
## modelos bayesianos:
prior = bExpo1(μ, α = αpriori, β = βpriori);
post = bExpo1(μ, xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpo1(μ, xobs = xx);
prior.familia
keys(post)
## estimación puntual de θ vía la mediana:
θ # valor teórico
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 
## estimación puntual predictiva vía la mediana: 
X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa
```
"""
function bExpo1(μ; xobs = zeros(0), α = 0.0, β = 0.0)
    n = length(xobs)
    zobs = xobs .- μ
    Z = bExpoEsc(xobs = zobs, α = α, β = β)  
    dpost(θ) = Z.d(θ)
    ppost(θ) = Z.p(θ)
    qpost(u) = Z.q(u)
    rpost(m) = Z.r(m)
    dpred(x) = Z.dp(x - μ)
    ppred(x) = Z.pp(x - μ)
    qpred(u) = μ + Z.qp(u)
    rpred(m) = μ .+ Z.rp(m) 
    return (familia = "Exponencial(μ,θ) con θ > 0 desconocido y μ = $μ", 
            d = dpost, p = ppost, q = qpost, r = rpost, dp = dpred, pp = ppred,
            qp = qpred, rp = rpred, n = n, muestra = xobs, α = α, β = β)
end


## Caso: μ ∈ ℜ desconocida, θ > 0 conocida
"""
    bExpo2(θ; xobs = zeros(0), α = 0, β = 0)

Distribuciones a posteriori y predictiva a posteriori para una distribución
Exponencial(μ,θ) donde μ ∈ ℜ es desconocido, pero θ > 0 es conocido, con
función de densidad de probabilidades:

```math 
f(x|μ) = θexp(-θ(x - μ)),  x > μ
```

donde:

- `θ` = valor conocido de este parámetro 
- `xobs` = vector con la muestra aleatoria observada (sin muestra, si no se especifica)
- `α, β` = hiperparámetros de la distribución a priori para λ:=θμ ~ π(λ|α,β) = βexp(β(λ-α)), λ<α

> Utiliza la función `bExpoLoc`

Entrega un tupla etiquetada con los siguientes elementos:

1. `familia` = distribución de probabilidad
2. `d` = función de densidad a posteriori para μ
3. `p` = función de distribución a posteriori para μ
4. `q` = función de cuantiles a posteriori para μ
5. `r` = función simuladora a posteriori para μ
6. `dp` = función de densidad predictiva a posteriori
7. `pp` = función de distribución predictiva a posteriori
8. `qp` = función de cuantiles predictiva a posteriori 
9. `rp` = función simuladora predictiva a posteriori 
10. `n` = tamaño de la muestra observada 
11. `muestra` = vector de la muestra observada
12. `α, β` = valores de los hiperparámetros a priori 

# Ejemplo
```
μ = -1.5; # valor teórico del parámetro desconocido
θ = 3.7; # valor del parámetro conocido
X = vaExponencial(μ,θ);
n = 10; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 10*θ, 0.06/θ # hiperparámetros a priori
## modelos bayesianos:
prior = bExpo2(θ, α = αpriori, β = βpriori);
post = bExpo2(θ, xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpo2(θ, xobs = xx);
prior.familia
keys(post)
## estimación puntual de μ vía la mediana:
μ # valor teórico
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 
## estimación puntual predictiva vía la mediana: 
X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa
```
"""
function bExpo2(θ; xobs = zeros(0), α = 0.0, β = 0.0)
    n = length(xobs)
    yobs = θ .* xobs
    Y = bExpoLoc(xobs = yobs, α = α, β = β)  
    dpost(μ) = θ * Y.d(θ * μ)
    ppost(μ) = Y.p(θ * μ)
    qpost(u) = Y.q(u) / θ
    rpost(m) = Y.r(m) ./ θ
    dpred(x) = θ * Y.dp(θ * x)
    ppred(x) = Y.pp(θ * x)
    qpred(u) = Y.qp(u) / θ
    rpred(m) = Y.rp(m) / θ
    return (familia = "Exponencial(μ,θ) con μ ∈ ℜ desconocido y θ = $θ", 
            d = dpost, p = ppost, q = qpost, r = rpost, dp = dpred, pp = ppred,
            qp = qpred, rp = rpred, n = n, muestra = xobs, α = α, β = β)
end



## Caso: μ ∈ ℜ desconocida, θ > 0 desconocida

# desarróllalo...


@info "bExpoEsc  bExpoLoc  bExpo1  bExpo2"
