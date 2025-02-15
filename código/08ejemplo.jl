### Ejemplo de familia de localización y escala
### Autor: Dr. Arturo Erdely
### Versión: 2025-02-15

#=
    Si Z es una variable aleatoria absolutamente continua con
    función de densidad f(z) y función de distribución F, y
    definimos la variable aleatoria X := μ + σZ con constantes
    μ ∈ ℜ y σ > 0, entonces la distribución de probabilidad
    resultante para X es una familia de localización y dispersión
    construida a partir de Z (denominada distirbución estándar)
    con parámetro de localización μ y parámetro de dispersión σ.

    En ocasiones resulta más conveniente la reparametrización
    θ = 1/σ, es decir X = μ + Z/θ, y a θ se le llama parámetro de
    precisión porque V(X)=1/θ² y a mayor valor de θ menor varianza,
    en contraste con V(X)=σ² donde a mayor valor de σ mayor varianza
    (y por ello se le llama parámetro de dispersión).

    Podemos construir familias de localización y escala a partir de
    modelos de variables aleatorias continuas que ya estén programados
    en el paquete `Distributions` o bien programarlos directamente.
=#


## Programando directamente
"""
    vaExponencial(μ::Real, θ::Real)

Distribución Exponencial con parámetro de localización μ ∈ ℜ 
y parámetro de precisión θ > 0, definida mediante la variable
aleatoria

```math
X = μ + Z/θ
```

donde `Z` es una variable aleatoria exponencial estándar, es decir
con función de densidad:

```math
f(z) = exp(-z),  z > 0
```

Por ejemplo, si se define `X = vaExponencial(3, 2)` entonces:

- `X.familia` = Familia de distribución de probabilidad
- `X.param` = tupla de parámetros
- `X.fdp(x)` = función de densidad de probabilidades evaluada en `x`
- `X.fda(x)` = función de distribución (acumulada) de probabilidades evaluada en `x`
- `X.ctl(u)` = función de cuantiles evaluada en un valor `0 ≤ u ≤ 1`
- `X.sim(n)` = vector de tamaño `n` con una muestra aleatoria simulada
- `X.soporte` = soporte (o rango) de la variable aleatoria
- `X.mediana` = mediana teórica
- `X.ric` = rango intercuartílico
- `X.moda` = moda teórica
- `X.media` = media teórica
- `X.varianza` = varianza teórica

## Ejemplo
```
X = vaExponencial(3, 2);
keys(X)
println(X.familia)
X.param
X.param.loc
println(X.soporte)
X.mediana
X.fda(X.mediana) # verificando la mediana
X.ctl(0.5) # verificando un cuantil
X.ric # rango intercuartílico
X.ctl(0.75) - X.ctl(0.25) # verificando rango intercuartílico
X.media, X.varianza
xsim = X.sim(10_000); # muestra aleatoria de tamaño 10,000
diff(sort(xsim)[[2499, 7500]])[1] # rango intercuartílico muestral
using Statistics # paquete de la biblioteca estándar de Julia (no requiere instalación previa)
median(xsim), mean(xsim), var(xsim) # mediana, media y varianza muestrales
```
"""
function vaExponencial(μ::Real, θ::Real)
    if θ ≤ 0
        error("El parámetro de precisión θ debe ser positivo")
        return nothing
    else
        fdp(x) = (x ≥ μ) * θ * exp(-θ*(x-μ))
        fda(x) = (x ≥ μ) * (1 - exp(-θ*(x-μ)))
        ctl(u) = 0 ≤ u ≤ 1 ? μ - log(1-u)/θ : NaN
        alea(n::Integer) = ctl.(rand(n))
    end
    soporte = ("[$μ , ∞[", μ, Inf)
    mediana = ctl(0.5)
    ric = ctl(0.75) - ctl(0.25)
    moda = μ
    media = μ + 1/θ
    varianza = (1/θ)^2
    return (familia = "Exponencial", param = (loc = μ, prec = θ), soporte = soporte,
            fdp = fdp, fda = fda, ctl = ctl, sim = alea, mediana = mediana,
            ric = ric, moda = moda, media = media, varianza = varianza
    )
end

μ = -1.5
θ = 3.7
Y = vaExponencial(μ, θ)
Y.familia
Y.param 
Y.soporte
Y.mediana
Y.fda(Y.mediana)
Y.ctl(0.5)
Y.media
Y.varianza


## Vía el paquete `Distributions`

using Distributions
function FamLocEsc(Z, μ, σ)
    X = μ + σ*Z
    return X
end
σ = 1 / θ
X = FamLocEsc(Exponential(1), μ, σ)
median(X)
cdf(X, median(X))
quantile(X, 0.5)
mean(X)
var(X)
