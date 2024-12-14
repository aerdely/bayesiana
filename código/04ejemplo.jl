### Ejemplo de simulación de variable aleatoria discreta
### Autor: Dr. Arturo Erdely
### versión: 11-agosto-2024


## Cargar paquetes y código externo necesarios:
begin
    using Distributions # previamente instalado
    include("02probestim.jl") # en misma carpeta
end

@doc Distributions # info general sobre el paquete


@doc Binomial # info sobre el modelo Binomial

X = Binomial(7, 0.2) # definir modelo Binomial
params(X) # tupla de parámetros
X.n # parámetro individual
X.p # parámetro individual

# media teórica
X.n * X.p
mean(X)

# varianza teórica
X.n * X.p * (1 - X.p)
var(X)

# valores mínimo y máximo teóricos
extrema(X)
minimum(X)
maximum(X)

# simular muestra aleatoria
muestra = rand(X, 10_000)
mean(muestra) # media muestral 
var(muestra) # varianza muestral 
minimum(muestra) # mínimo muestral
maximum(muestra) # máximo muestral

# función de masa de probabilidades (fmp) teórica
pdf(X, 7)
pdf(X, [0,1])

# fmp teórica versus muestral
@doc masaprob 

Xm = masaprob(muestra);
xx = collect(minimum(X):maximum(X))
fmpTeórica = pdf(X, xx)
fmpMuestral = Xm.fmp.(xx)
hcat(xx, fmpTeórica, fmpMuestral)
