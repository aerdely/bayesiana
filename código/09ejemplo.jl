### Ejemplo muestreador de Gibbs
### Autor: Dr. Arturo Erdely
### Versión: 2025-03-30

#=
    (X,N) vector aleatorio tal que:

    X|N=n ~ Uniforme{0,...,n}
    N ~ Uniforme{1,2,3}
=#


## Paquetes y código necesarios

using Distributions, Random
include("02probestim.jl")



## Modelo teórico

# condicional P(X = x | N = n)

function pXcN(x::Int, n::Int)
    if !(n ∈ [1,2,3])
        return NaN 
    end
    if 0 ≤ x ≤ n 
        return 1/(n+1)
    else
        return 0.0
    end
end

function rXcN(nsim::Int, n::Int) # simulador condicional X|N=n
    if !(n ∈ [1,2,3]) || nsim ≤ 0 
        return Int[]
    else
        return rand(0:n, nsim)
    end
end

# probando el simulador condicional X|N=n
println(rXcN(20, 1))
masaprob(rXcN(100_000, 1)).probs
println(rXcN(20, 2))
masaprob(rXcN(100_000, 2)).probs
println(rXcN(20, 3))
masaprob(rXcN(100_000, 3)).probs
println(rXcN(20, 4))


# marginal P(N = n)

function pN(n::Int)
    if n ∈ [1,2,3]
        return 1/3 
    else
        return 0.0 
    end
end


# conjunta P(X = x , N = n)

pXN(x::Int, n::Int) = pXcN(x, n) * pN(n) 

pXN_mat = fill(NaN, 3, 4) # colocar valores en una matriz
for n ∈ 1:3
    for x ∈ 0:3
        pXN_mat[n, x+1] = pXN(x, n)
    end
end
pXN_mat
sum(pXN_mat)
sum(pXN_mat, dims=2) # marginal de N
sum(pXN_mat, dims=1) # marginal de X 


# marginal P(X = x)

function pX(x::Int)
    if !(x ∈ [0,1,2,3])
        return 0.0
    else
        p = 0.0
        for n ∈ [1,2,3]
            p += pXN(x, n)
        end
        return p
    end    
end
pX.([0,1,2,3])


# condicional P(N = n | X = x)

function pNcX(n::Int, x::Int)
    if !(x ∈ [0,1,2,3])
        return NaN
    end
    if n ∈ [1,2,3]
        return pXN(x, n) / pX(x)
    else
        return 0.0
    end
end

pNcX_mat = fill(NaN, 3, 4) # colocar valores en una matriz
for n ∈ 1:3
    for x ∈ 0:3
        pNcX_mat[n, x+1] = pNcX(n, x)
    end
end
pNcX_mat
sum(pNcX_mat, dims=1)

function rNcX(nsim::Int, x::Int) # simulador condicional N|X=x
    if nsim ≤ 0 || !(x ∈ [0,1,2,3])
        return Int[]
    else
        p = pNcX.([1,2,3], x)
        return rand(Categorical(p), nsim)
    end
end 

begin # probando simulador condicional N|X=x
    x = 0 # x ∈ {0,1,2,3}
    nval = [1,2,3]
    prob = pNcX.(nval, x)
    sim = masaprob(rNcX(100_000, x))
    pemp = sim.fmp.(nval)
    [nval prob pemp]
end



## Muestreador de Gibbs 

# Partiendo de X 

begin # generar cadenas de Markov
    x = 0 # valor inicial: x ∈ {0,1,2,3}
    nsim = 100_000
    N = zeros(Int, nsim)
    X = zeros(Int, nsim)
    for k ∈ 1:nsim 
        N[k] = rNcX(1, x)[1]
        X[k] = rXcN(1, N[k])[1]
        x = X[k]
    end
end

function analizar(cadena::Vector{<:Real}, posiciones::Vector{<:Int})
    # estimar empíricamente fmp marginal
    estim = masaprob(cadena[posiciones])
    return [estim.valores estim.probs]
end

analizar(N, collect(1:1000))

function sinreemplazo(A::Array, m::Integer)
    # muestra sin reemplazo de tamaño m a partir de n elementos (m ≤ n)
    # requiere haber ejectado primero: using Random
    if m > length(A)
        error("m debe ser igual o menor que el número de elementos de A")
        return nothing
    else
        return shuffle(A)[1:m]
    end
end

sinreemplazo(collect('a':'z'), 10) # ejemplo de uso

begin # análisis marginal de N 
    nval = [1,2,3]
    pNteórica = pN.(nval)
    pNestim1 = analizar(N, collect(1:1000))[:, 2]
    pNestim2 = analizar(N, collect(99_001:100_000))[:, 2]
    pNestim3 = analizar(N, sinreemplazo(collect(10_000:100_000), 1000))[:, 2]
    [nval pNteórica pNestim1 pNestim2 pNestim3]
end

begin # análisis marginal de X 
    xval = [0,1,2,3]
    pXteórica = pX.(xval)
    pXestim1 = analizar(X, collect(1:1000))[:, 2]
    pXestim2 = analizar(X, collect(99_001:100_000))[:, 2]
    pXestim3 = analizar(X, sinreemplazo(collect(10_000:100_000), 1000))[:, 2]
    [xval pXteórica pXestim1 pXestim2 pXestim3]
end


# Partiendo de N 

begin # generar cadenas de Markov
    n = 1 # n ∈ {1,2,3}
    nsim = 100_000
    N = zeros(Int, nsim)
    X = zeros(Int, nsim)
    for k ∈ 1:nsim 
        X[k] = rXcN(1, n)[1]
        N[k] = rNcX(1, X[k])[1]
        n = N[k]
    end
end

begin # análisis marginal de N 
    nval = [1,2,3]
    pNteórica = pN.(nval)
    pNestim1 = analizar(N, collect(1:1_000))[:, 2]
    pNestim2 = analizar(N, collect(99_001:100_000))[:, 2]
    pNestim3 = analizar(N, sinreemplazo(collect(10_000:100_000), 1000))[:, 2]
    [nval pNteórica pNestim1 pNestim2 pNestim3]
end

begin # análisis marginal de X
    xval = [0,1,2,3]
    pXteórica = pX.(xval)
    pXestim1 = analizar(X, collect(1:1_000))[:, 2]
    pXestim2 = analizar(X, collect(99_001:100_000))[:, 2]
    pXestim3 = analizar(X, sinreemplazo(collect(10_000:100_000), 1000))[:, 2]
    [xval pXteórica pXestim1 pXestim2 pXestim3]
end


# Análisis conjunto (X,N)

pXN_mat # valores teóricos P(X = x , N = n) ≡ pXN_mat[n, x+1]

function analizar2(X::Vector{<:Real}, N::Vector{<:Real}, posiciones::Vector{<:Int})
    # estimar empíricamente la fmp conjunta
    Δ = 1 / length(posiciones)
    p = zeros(3, 4)
    for k ∈ eachindex(posiciones)
        p[N[posiciones[k]], X[posiciones[k]]+1] += Δ
    end
    return p
end

analizar2(X, N, collect(1:10_000))
analizar2(X, N, collect(90_001:100_000))
analizar2(X, N, sinreemplazo(collect(10_000:100_000), 10_000))
