### Función de masa de probabilidades
### Autor: Dr. Arturo Erdely
### versión: 11-agosto-2024

### Funciones:  simDiscreta  discretaFinita


"""
    simDiscreta(fmp::Function, m::Integer)

Simulador de una muestra aleatoria tamaño `m` a partir de una función
de masa de probabilidades `fmp`.

## Ejemplo
```
f(k) = 6 / (π*k)^2 # k ∈ {1,2,3,...}
f(1), f(2), f(3)
sum(f.(collect(1:1_000_000))) # suma parcial cercana a 1 por la izquierda
m = 1_000_000; # tamaño de muestra a simular
s = simDiscreta(f, m) # vector de simulaciones
prop(k) = count(s .== k) / m # probabilidad empírica
# Algunas probabilidades teóricas versus empíricas:
f(1), prop(1)
f(2), prop(2)
f(3), prop(3)
f(7), prop(7)
```
"""
function simDiscreta(fmp::Function, m::Integer)
    # fmp(k) = probabilidad de k ∈ ℵ = {1,2,3,...}
    function simuno(u)
        k = 1
        a = fmp(1)
        while u > a
            k += 1
            a += fmp(k)
        end
        return k
    end
    return simuno.(rand(m))
end


"""
    discretaFinita(Ω, p)

Distribución de probabilidad sobre el vector `Ω` con probabilidades `p`.
Crea una tupla con los siguientes elementos:
- `nom` = nombre de la ley de probabilidades
- `valores` = vector de valores posibles `Ω`
- `probs` =  vector de probabilidades `p` para los elementos de `Ω`
- `fmp` = función de masa de probabilidades tal que `fmp(x)` es la probabilidad de `x`
- `sim` = simulador de valores de `Ω` con probabilidades `p` tal que `sim(n)` simula una muestra tamaño `n`

## Ejemplo
```
D = discretaFinita(['a', 'b', 'c'], [0.1, 0.7, 0.2]);
keys(D)
D.nom
D.valores
D.probs
D.fmp('b')
D.sim(5)
count(D.sim(1_000_000) .== 'b')
```
"""
function discretaFinita(Ω, p)
    if !(isa(Ω, Vector) && isa(p, Vector) && length(Ω) == length(p))
        @error "Ambos argumentos deben ser vectores de la misma longitud"
        return nothing
    end
    if !all(p .≥ 0)
        @error "Los valores del vector `p` no pueden ser negativos"
        return nothing
    elseif sum(p) ≠ 1.0
        p = p ./ sum(p)
        @warn "Los valores de `p` fueron forzados a sumar 1"
    end
    nombre = "Distribución discreta finita"
    pacum = cumsum(p) # partición del intervalo [0,1]
    function simuno(u) # mapea a 0<u<1 a la partición que le corresponde
        c = 1
        for p ∈ pacum
            c += p < u
        end
        return c 
    end
    simular(n) = Ω[simuno.(rand(n))] # simulador
    d = Dict(Ω .=> p) # función de masa de probabilidades (fmp)
    function prob(x)
        if x ∈ Ω
            return d[x]
        else
            return 0.0
        end
    end
    return (nom = nombre, valores = Ω, probs = p, fmp = prob, sim = simular)
end

@info "simDiscreta  discretaFinita"
