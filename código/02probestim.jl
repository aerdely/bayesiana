### Estimar distribuciones de probabilidades
### Autor: Dr. Arturo Erdely
### versión: 10-agosto-2024

### Funciones:  distprob  masaprob  densprob


"""
    distprob(muestra::Vector{<:Real})

Estima la función de distribución acumulativa `F(x) = P(X ≤ x)` de una
variable aleatoria `X` a partir de un vector `muestra` de observaciones de `X`.
Entrega una tupla etiquetada con los siguientes nombres:
- `fda` la función de distribución acumulativa estimada
- `min`, `max` mínimo y máximo muestrales
- `mord` la muestra ordenada de menor a mayor

## Ejemplo
```
muestra = randn(10_000);
D = distprob(muestra);
D.min, D.max
D.mord
F = D.fda;
F(0)
diff(F.([-1.96, 1.96]))
```
"""
function distprob(muestra::Vector{<:Real})
    Fn(x) = count(muestra .≤ x) / length(muestra)
    inf, sup = extrema(muestra) # mínimo y máximo muestrales
    return (fda = Fn, min = inf, max = sup, mord = sort(muestra))
end


"""
    masaprob(muestra::Vector)

Estima la función de masa de probabilidades a partir del vector `muestra`.
Entrega una tupla etiquetada con los siguientes nombres:
- `fmp` una función que estima la probabilidad de su argumento
- vector con los `valores` distintos encontrados en la `muestra`
- `probs` vector de proporciones muestrales de los `valores`
- la `muestra` proporcionada

## Ejemplo
```
muestra = rand(['a','b','c'], 100);
M = masaprob(muestra);
hcat(M.valores, M.probs) # tabla
M.fmp('c')
M.fmp.(['a', 'b'])
sum(M.fmp.(M.valores)) # deber ser 1.0 (aprox)
```
"""
function masaprob(muestra::Vector)
    valores = unique(muestra) # valores distintos de la muestra
    try 
        sort!(valores) # intenta ordenar valores
    catch
        @warn "Conjunto no ordenable" # avisa si no se pudo
    end 
    nv = length(valores) # número de valores distintos en la muestra
    frecuencia = zeros(Int, nv) # vector de frecuencias absolutas
    n = length(muestra) # tamaño de la muestra
    for i ∈ 1:nv # calcular frecuencia absoluta de los distintos valores de la muestra
        for j ∈ 1:n
            if valores[i] == muestra[j]
                frecuencia[i] += 1
            end
        end
    end
    proporción = frecuencia ./ n
    d = Dict(valores .=> proporción)
    function f(x)
        if x ∈ valores
            return d[x]
        else
            return 0.0
        end
    end
    return (fmp = f, valores = valores, probs = proporción, muestra = muestra)
end


"""
    densprob(muestra::Vector{<:Real}, nclases::Integer = 0)

Estima la función de densidad correspondiente al vector `muestra` de observaciones
de una variable aleatoria absolutamente continua, mediante una función constante
por intervalos, considerando un total de `nclases` intervalos, parámetro que de
omitirse se calcula como el `mín{√n, 30}` donde `n` es el tamaño de muestra.
Entrega una tupla etiquetada con los siguientes nombres:
- `fdp` es la función de densidad estimada
- `clases` es un vector que particiona el rango observado en `nclases`
- `reps` es un vector de puntos medios de las clases (representantes)
- `min` y `max` son el mínimo y máximo muestrales
- `nclases` el número de clases o intervalos de igual longitud
- y la `muestra` utilizada

## Ejemplo
```
muestra = randn(10_000); # Normal(0,1)
D = densprob(muestra, 25);
D.fdp(0)
D.fdp.([-2,-1,0,1,2])
D.min, D.max
D.clases
D.reps
D.nclases
sum(D.fdp.(D.reps)) * (D.max - D.min)/D.nclases # debe ser 1.0 (aprox)
D.muestra
```
"""
function densprob(muestra::Vector{<:Real}, nclases::Integer = 0)
    # muestra = vector de valores observados unidimensionales
    # numint = número de intervalos (clases) a considerar
    n = length(muestra) # tamaño de muestra
    if nclases == 0 # valor por defecto
        m = min(Int(round(sqrt(n))), 30)
    else
        m = nclases
    end    
    inf, sup = extrema(muestra) # mínimo y máximo muestrales
    clases = collect(range(inf, sup, length = m + 1)) # extremos de intervalos de clase
    repsclase = (clases[1:end-1] + clases[2:end])/2 # puntos medios de las clases
    function f(x::Real) # función de densidad estimada
        if x < inf || x > sup
            return 0.0 # por estar fuera del rango observado
        else
            i = count(x .≥ clases) # determinar clase a la que pertenece x
            if i ∈ [m, m+1] # último intervalo es de la forma [a,b]
                a = clases[m]
                b = clases[m+1]
                conteo = count(a .≤ muestra .≤ b)
            else # los demás intervalos son de la forma [a,b)
                a = clases[i]
                b = clases[i+1]
                conteo = count(a .≤ muestra .< b)
            end
            return conteo / (n*(b-a))
        end
    end
    return (fdp = f, clases = clases, reps = repsclase, min = inf, max = sup, nclases = m, muestra = muestra)
end

@info "distprob  masaprob  densprob"
