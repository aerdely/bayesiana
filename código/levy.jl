### Distancia de Lévy
### Autor: Dr. Arturo Erdely
### Fecha: 2026-10-24

#=
    Distancia de Lévy entre dos funciones de
    distribución acumulativa univariadas.
=#

"""
    function levy_puntual(x::Real, F::Function, G::Function; tol = 1e-8, maxiter = 1_000)

Aproxima la distancia de Lévy entre las funciones de distribución acumulativas univariadas `F` y `G` en el punto `x`, con un nivel de tolerancia `tol` en la aproximación, y un máximo de `maxiter` iteraciones.

## Ejemplo
```
F(x) = 1 - exp(-x) # Exponencial(1)
G(x) = 1 - inv(x)  # Pareto(1,1)
levy_puntual(1, F, G)
levy_puntual(2, F, G)
levy_puntual(10, F, G)  
```
"""
function levy_puntual(x::Real, F::Function, G::Function;
                      tol = 1e-8, maxiter = 1_000)

    0 < tol < 1 || throw(ArgumentError("`tol` debe ser un valor entre 0 y 1"))
    maxiter > 0 || throw(ArgumentError("`maxiter` debe ser un entero positivo"))

    c = F(x)

    cumple(δ) = G(x - δ) - δ <= c <= G(x + δ) + δ

    cumple(0.0) && return 0.0

    a, b = 0.0, 1.0
    it = 0

    for iter in 1:maxiter
	it += 1
        b - a <= tol && return b
        δ = (a + b) / 2

        if δ == a || δ == b
            @warn "Se alcanzó el límite de precisión por redondeo"
            return b
        end

        if cumple(δ)
            b = δ
        else
            a = δ
        end
    end

    b - a > tol && @warn "No se alcanzó el nivel $(tol) de tolerancia"
    it = maxiter && @warn "Se alcanzó el número máximo de $(maxiter) iteraciones"
    return b
end


#=
    Estimación empírica con base en dos muestras univariadas.
=#

using Random
using Statistics

"""Verifica d_L(F_n,G_m) <= delta, para muestras ordenadas."""
function _levy_factible(x::Vector{Float64}, y::Vector{Float64}, delta::Float64)
    n, m = length(x), length(y)
    invn, invm = 1.0 / n, 1.0 / m

    j = 0
    @inbounds for i in 1:n
        limite = x[i] + delta
        while j < m && y[j + 1] <= limite
            j += 1
        end
        i * invn <= j * invm + delta || return false
    end

    i = 0
    @inbounds for j in 1:m
        limite = y[j] + delta
        while i < n && x[i + 1] <= limite
            i += 1
        end
        j * invm <= i * invn + delta || return false
    end

    return true
end


"""Calcula la distancia de Kolmogórov para muestras ordenadas."""
function _kolmogorov_ordenadas(x::Vector{Float64}, y::Vector{Float64})
    n, m = length(x), length(y)
    invn, invm = 1.0 / n, 1.0 / m
    i = j = 0
    d = 0.0

    @inbounds while i < n || j < m
        t = if i == n
            y[j + 1]
        elseif j == m
            x[i + 1]
        else
            min(x[i + 1], y[j + 1])
        end
        # Los empates se procesan simultáneamente.
        while i < n && x[i + 1] == t
            i += 1
        end
        while j < m && y[j + 1] == t
            j += 1
        end
        d = max(d, abs(i * invn - j * invm))
    end
    return d
end

"""Distancia de Lévy para muestras ya ordenadas y Float64."""
function _levy_ordenadas(x::Vector{Float64}, y::Vector{Float64}, tol::Float64)
    _levy_factible(x, y, 0.0) && return 0.0

    a = 0.0
    # d_L <= d_K. nextfloat evita problemas de redondeo en la cota.
    b = min(1.0, nextfloat(_kolmogorov_ordenadas(x, y)))
    # Respaldo defensivo frente a errores de redondeo.
    _levy_factible(x, y, b) || (b = 1.0)

    while b - a > tol
        delta = a + (b - a) / 2
        (delta == a || delta == b) && break
        if _levy_factible(x, y, delta)
            b = delta
        else
            a = delta
        end
    end
    return b   # cota superior factible, dentro de tol del ínfimo
end


"""
    distancia_levy_empirica(xobs, yobs; tol=1e-8)

Calcula d_L entre las FDA empíricas. 
"""
function distancia_levy_empirica(xobs::AbstractVector{<:Real},
                                yobs::AbstractVector{<:Real};
                                tol::Real = 1e-8)
    0 < tol < 1 || throw(ArgumentError("tol debe estar entre 0 y 1"))
    isempty(xobs) && throw(ArgumentError("xobs no puede estar vacía"))
    isempty(yobs) && throw(ArgumentError("yobs no puede estar vacía"))
    x = Float64.(xobs)
    y = Float64.(yobs)
    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("Las muestras deben contener números finitos"))
    sort!(x)
    sort!(y)
    return _levy_ordenadas(x, y, Float64(tol))
end


"""
    intervalo(muestra::Vector{<:Real}, prob::Real = 0.95)

Intervalo de amplitud minima que cubre por lo menos `prob` de los
valores de la muestra (contando observaciones, no valores distintos).
No es un intervalo de confianza para la distancia verdadera.
"""
function intervalo(muestra::AbstractVector{<:Real}, prob::Real = 0.95)
    isempty(muestra) && throw(ArgumentError("muestra vacia"))
    0 < prob <= 1 || throw(ArgumentError("prob debe estar en (0,1]"))
    xx = sort(Float64.(muestra))
    all(isfinite, xx) || throw(ArgumentError("muestra no finita"))
    n = length(xx)
    k = ceil(Int, prob * n)
    a, b = xx[1], xx[k]
    minimo = b - a
    @inbounds for i in 2:(n-k+1)
        ancho = xx[i+k-1] - xx[i]
        if ancho < minimo
            minimo = ancho
            a, b = xx[i], xx[i+k-1]
        end
    end
    proporcion_real = (searchsortedlast(xx, b) - searchsortedfirst(xx, a) + 1) / n
    return (int = [a, b], tam = minimo, prob = proporcion_real,
            minmax = [xx[1], xx[end]], ran = xx[end] - xx[1])
end


"""
    levy_empirica(xobs, yobs; nsim=1000, tol=1e-8, rng=Random.default_rng())

Distancia empírica y, opcionalmente, distribucion simulada bajo la
hipotesis de una distribucion comun usando bootstrap del conjunto combinado.
"""
function levy_empirica(xobs::AbstractVector{<:Real},
                       yobs::AbstractVector{<:Real};
                       nsim::Int = 1_000,
                       tol::Real = 1e-8,
                       rng::AbstractRNG = Random.default_rng())
    nsim >= 0 || throw(ArgumentError("nsim debe ser no negativo"))
    # Validar y ordenar una sola vez.
    x = Float64.(xobs)
    y = Float64.(yobs)
    0 < tol < 1 || throw(ArgumentError("tol debe estar entre 0 y 1"))
    isempty(x) && throw(ArgumentError("xobs no puede estar vacía"))
    isempty(y) && throw(ArgumentError("yobs no puede estar vacía"))
    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("Las muestras deben contener números finitos"))
    xy = vcat(x, y)
    sort!(x)
    sort!(y)
    dL = _levy_ordenadas(x, y, Float64(tol))
    nsim == 0 && return (dist = dL,)

    nx, ny = length(x), length(y)
    xsim = Vector{Float64}(undef, nx)
    ysim = Vector{Float64}(undef, ny)
    dLsim = Vector{Float64}(undef, nsim)

    for s in eachindex(dLsim)
        @inbounds for i in eachindex(xsim)
            xsim[i] = rand(rng, xy)
        end
        @inbounds for j in eachindex(ysim)
            ysim[j] = rand(rng, xy)
        end
        sort!(xsim)
        sort!(ysim)
        dLsim[s] = _levy_ordenadas(xsim, ysim, Float64(tol))
    end

    p_boot = (1 + count(v -> v >= dL, dLsim)) / (nsim + 1)
    return (dist = dL,
            Ho = median(dLsim),
            int95 = intervalo(dLsim, 0.95).int,
            int99 = intervalo(dLsim, 0.99).int,
            q95 = quantile(dLsim, 0.95),
            q99 = quantile(dLsim, 0.99),
            p_boot = p_boot,
            rango = extrema(dLsim),
            simHo = dLsim)
end
