### Ejemplo de estimación por máxima verosimilitud
### Autor: Dr. Arturo Erdely
### Versión: 2024-08-11


# Cargar paquetes y código externo necesarios:
begin
    using Distributions # previamente instalado
    include("06EDA.jl") # en misma carpeta
end


# Simulamos una muestra aleatoria
begin
    m, p = 17, 0.35 # parámetros
    X = Binomial(m, p) # variable aleatoria Binomial
    n = 100 # tamaño de muestra a simular
    muestra = rand(X, n)
    println(muestra)
end

# Estimación inicial por método de momentos
begin
    μ = mean(muestra)
    σ2 = var(muestra)
    p0 = 1 - σ2/μ
    m0 = max(1, Int(round(μ/p0)))
    m0, p0
end

# Estimación por máxima verosimilitud
mlogV(θ) = -sum(log.(pdf(Binomial(θ[1], θ[2]), muestra))) # menos log-verosimilitud
mlogV([m, p])
mlogV([15, 0.6])
emv = EDA(mlogV, [10, 0.15], [20, 0.6], iEnteros = [1])
