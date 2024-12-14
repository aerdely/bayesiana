### Ejemplo de simulación de variables aleatorias continuas
### Autor: Dr. Arturo Erdely
### versión: 11-agosto-2024


## Cargar paquetes y código externo necesarios:
begin
    using Distributions, Plots # previamente instalados
    include("02probestim.jl") # en misma carpeta
end


@doc Gamma

X = Gamma(2,3)

α, θ = params(X)

mean(X), α*θ # media teórica
var(X), α*(θ^2) # varianza teórica
extrema(X) # soporte

m = median(X) # mediana teórica
cdf(X, m) # P(X ≤ m) = 1/2
quantile(X, 1/2) # cuantil 1/2 = mediana 


# muestra simulada
muestra = rand(X, 1_000)
mean(muestra), var(muestra)
median(muestra)
quantile(muestra, 1/2)


# Función de distribución empírica versus teórica
@doc distprob

D = distprob(muestra);
keys(D)
xx = collect(range(0.0, D.max, length = 1_000))
plot(xx, D.fda.(xx), lw = 3, label = "empírica", color = :blue,
     xlabel = "x", ylabel = "F(x) = P(X ≤ x)",
     title = "Función de distribución de probabilidades"
)
plot!(xx, cdf(X, xx), lw = 2, color = :red, label = "teórica")
savefig("05ejemploA.pdf")


# Función de densidad empírica versus teórica
@doc densprob

d = densprob(muestra);
plot(xx, d.fdp.(xx), lw = 3, label = "empírica", color = :blue,
     xlabel = "x", ylabel = "f(x)",
     title = "Función de densidad de probabilidades"
)
plot!(xx, pdf(X, xx), lw = 2, color = :red, label = "teórica")
savefig("05ejemploB.pdf")
