#=
    Exponencial(μ,θ)    μ ∈ ℜ    θ > 0
=#

using Distributions

## Caso: θ > 0, μ = 0
"""
    bExpoEsc(; xobs = zeros(0), α = 0, β = 0)

Distribuciones a posteriori y predictiva a posteriori para una distribución
Exponencial(μ=0, θ>0) donde θ es parámetro de escala:
```math 
f(x) = θexp(-θx), x > 0
```
donde:

- `xobs` = vector con la muestra aleatoria observada (por defecto: sin muestra)
- `α, β` = hiperparámetros de la distribución a priori Gamma(α,β)

> Requiere que el paquete `Distributions` sea previamente instalado.

Entrega un tupla etiquetada con los siguientes elementos:

1. `familia` = distribución de probabilidad
2. `conjugada` = distribución a posteriori de θ (objeto del paquete `Distributions`)
3. `d` = función de densidad a posteriori para θ
4. `p` = función de distribución a posteriori para θ
5. `q` = función de cuantiles a posteriori para θ
6. `r` = función simuladora a posteriori para θ
7. `dp` = función de densidad predictiva a posteriori
8. `pp` = función de distribución predictiva a posteriori
9. `qp` = función de cuantiles predictiva a posteriori 
10. `rp` = función simuladora predictiva a posteriori 
11. `n` = tamaño de la muestra observada 
12. `sx` = suma de la muestra observada 
13. `muestra` = vector de la muestra observada
14. `α, β` = valores de los hiperparámetros a priori 

# Ejemplo
```
θ = 3.7;
X = Exponential(1/θ);
# se usa 1/θ por la reparametrización 
# utilizada en el paquete `Distributions`
n = 100; # tamaño de muestra a simular
xx = rand(X, n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori
## modelos bayesianos:
prior = bExpoEsc(α = αpriori, β = βpriori);
post = bExpoEsc(xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpoEsc(xobs = xx);
prior.familia
prior.conjugada
post.conjugada
postNoinfo.conjugada 
keys(post)
## estimación puntual de θ vía la mediana:
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 
## estimación puntual predictiva vía la mediana: 
median(X) # teórica 
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
    rpred(n) = qpred.(rand(n))
    return (familia = "Exponencial de escala θ>0", conjugada = post, d = dpost, p = ppost,
            q = qpost, r = rpost, dp = dpred, pp = ppred, qp = qpred, rp = rpred,
            n = n, sx = ∑xᵢ, muestra = xobs, α = α, β = β)
end


## Caso: μ ∈ ℜ, θ = 1

# … desarróllalo


## Caso: μ ∈ ℜ, θ > 0

# … desarróllalo
