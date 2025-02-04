# Ejercicio 1.1

#= 
    Considere una muestra aleatoria X₁,…,Xₙ ~ Bernoulli(θ)
    con parámetro desconocido 0 < θ < 1, así como información
    a priori que se traduce en que Mediana(θ) = 0.36, y en que
    P(θ ≤ 0.5) = 0.95.

    a) Deduzca los hiperparámetros α > 0 y β > 0 para una 
       distribución a priori Beta(α,β) que refleje la
       información a priori.
    b) Deduzca los hiperparámetros a > 0 y b > 0 para una 
       distribución a priori Kumaraswamy(a,b) que refleje la
       información a priori.
    c) En una misma gráfica compare ambas funciones de densidad
       a priori.

    Ahora considere que se obtuvo la muestra aleatoria observada {1,0,0,1,1,0,1,0,0,1}.

    d) Obtenga la distribución a posteriori de θ utilizando como
       distribución a priori la obtenida en el inciso a).
    e) Obtenga la distribución a posteriori de θ utilizando como
       distribución a priori la obtenida en el inciso b).
    f) En una misma gráfica compare las funciones de densidad
       a posteriori obtenidas en los incisos d) y e).
    g) En una misma gráfica compare las funciones de densidad
       Beta a priori, a posteriori y a posteriori no informativa.
    h) Con base en la distribución a posteriori obtenida en el inciso d)
       obtenga una estimación por intervalo de probabilidad 0.99 y de
       longitud mínima para θ, y compáralo versus un intervalo [c,d] tal que
       P(θ ≤ c) = 0.005 = P(θ ≥ d).
    i) Ahora con base en una muestra aleatoria observada de tamaño n = 100
       donde ∑xᵢ = 50 repita lo realizado en los incisos d) y g)
=#


## Paquetes y código externo necesarios
begin
   using Distributions, Optim, QuadGK
   using Plots, LaTeXStrings
   include("06EDA.jl")
end


## a) A priori Beta 

function hBeta(p)
    if p[1] ≤ 0.0 || p[2] ≤ 0.0
        return Inf 
    else
        B = Beta(p[1], p[2])
        c1 = median(B) - 0.36
        c2 = cdf(B, 0.5) - 0.95
        return c1^2 + c2^2
    end
end

# utilizando algoritmo de EDA.jl
solBetaEDA = EDA(hBeta, [0.0001, 0.0001], [10.0, 10.0], tamgen = 10_000)
priorBetaEDA = Beta(solBetaEDA.x[1], solBetaEDA.x[2])
# comprobando
median(priorBetaEDA)
cdf(priorBetaEDA, 0.5)

# utilizando paquete Optim 
solBeta = optimize(hBeta, [1,1]) # ERROR
solBeta = optimize(hBeta, [10,10]) # ERROR
solBeta = optimize(hBeta, solBetaEDA.x) # aprovechar la pseudo-solución de EDA
pp = Optim.minimizer(solBeta)
priorBeta = Beta(pp[1], pp[2])
# comprobando
median(priorBeta)
cdf(priorBeta, 0.5)


## b) A priori Kumaraswamy

function hKuma(p)
   if p[1] ≤ 0.0 || p[2] ≤ 0.0
       return Inf 
   else
       K = Kumaraswamy(p[1], p[2])
       c1 = median(K) - 0.36
       c2 = cdf(K, 0.5) - 0.95
       return c1^2 + c2^2
   end
end

# pseudo-solución vía EDA
solKumaEDA = EDA(hKuma, [0.0001, 0.0001], [10.0, 10.0], tamgen = 10_000)
priorKumaEDA = Kumaraswamy(solKumaEDA.x[1], solKumaEDA.x[2])
# comprobando
median(priorKumaEDA)
cdf(priorKumaEDA, 0.5)
# mejorar pseudo-solución EDA con Optim 
solKuma = optimize(hKuma, solKumaEDA.x)
qq = Optim.minimizer(solKuma)
priorKuma = Kumaraswamy(qq[1], qq[2])
# comprobando
median(priorKuma)
cdf(priorKuma, 0.5)


## c) Comparando distribuciones a priori

begin
   θ = range(0, 1, length = 1_000)
   πBeta = pdf(priorBeta, θ)
   πKuma = pdf(priorKuma, θ)
   plot(θ, πBeta, lw = 3, label = "Beta", title = "Densidades a priori")
   xaxis!(L"\theta")
   yaxis!(L"\pi(\theta)")
   plot!(θ, πKuma, lw = 3, label = "Kumaraswamy")
   vline!([0.36], label = "mediana")
   vline!([0.5], label = "cuantil 0.95")
end


## d) A posteriori Beta 
begin
   xobs = [1,0,0,1,1,0,1,0,0,1]
   n = length(xobs)
   sx = sum(xobs)
   postBeta = Beta(priorBeta.α + sx, priorBeta.β + n - sx)
   yBetaPost = pdf(postBeta, θ)
end;


## e) A posteriori con a priori Kumaraswamy

L(θ) = (0 < θ < 1) * (θ^sx) * ((1-θ)^(n-sx)) # verosimilitud
π(θ) = pdf(priorKuma, θ)
g(θ) = L(θ)*π(θ)
# integración numérica con paquete QuadGK
c = quadgk(g, 0, 1)
postKuma(θ) = g(θ) / c[1]
quadgk(postKuma, 0, 1)
yKumaPost = postKuma.(θ);


## f) A posteriori con a prioris Beta vs Kumaraswamy
begin
   plot(θ, yBetaPost, lw = 3, label = "Con Beta a priori")
   plot!(θ, yKumaPost, lw = 3, label = "Con Kumaraswamy a priori")
   title!("Densidades a posteriori")
   xaxis!(L"\theta")
   yaxis!(L"p(\,\theta\,\,|\,\mathbf{x}_{obs})")
end


## g) Beta a priori, a posteriori y posteriori no informativa
begin
   plot(θ, πBeta, lw = 3, label = "a priori Beta")
   xaxis!(L"\theta")
   yaxis!("densidad")
   plot!(θ, yBetaPost, lw = 3, label = "a posteriori Beta, n = $n")
   postBetaNoinfo = Beta(0.001 + sx, 0.001 + n - sx)
   yBetaPostNoinfo = pdf(postBetaNoinfo, θ)
   plot!(θ, yBetaPostNoinfo, lw = 3, label = "a posteriori Beta no informativa")
end


## h) Estimación por intervalo.

γ = 0.99 # probabilidad del intervalo 

# estimación no óptima 
c = quantile(postBeta, (1-γ)/2)
d = quantile(postBeta, (1+γ)/2)
cdf(postBeta, d) - cdf(postBeta, c)
cdf(postBeta, c)
1 - cdf(postBeta, d)
d - c # longitud del intervalo

# estimación óptima
b(a) = quantile(postBeta, γ + cdf(postBeta, a)) # extremo superior en función del inferior
# longitud del intervalo [a,b] a minimizar
function ℓ(a)
   if 0.0 ≤ a[1] ≤ quantile(postBeta, 1-γ)
      return b(a[1]) - a[1]   
   else 
      return Inf
   end
end
# comprobando el caso no óptimo
b(c)
ℓ(c)
# minimzando longitud de intervalo con paquete Optim
Iopt1 = optimize(ℓ, [quantile(postBeta, 1-γ)/2])
Optim.minimizer(Iopt1)
# comparando con EDA
Iopt2 = EDA(ℓ, [0], [quantile(postBeta, 1-γ)])
# intervalo óptimo
aopt = Optim.minimizer(Iopt1)[1]
bopt = b(aopt)
ℓ(aopt) # longitud mínima


## i) Repetir d) y g) pero con n = 100 y ∑xᵢ=50

begin
   n = 100
   sx = 50
   postBeta = Beta(priorBeta.α + sx, priorBeta.β + n - sx)
   yBetaPost = pdf(postBeta, θ)
   plot(θ, πBeta, lw = 3, label = "a priori Beta")
   xaxis!(L"\theta")
   yaxis!("densidad")
   plot!(θ, yBetaPost, lw = 3, label = "a posteriori Beta, n = $n")
   postBetaNoinfo = Beta(0.001 + sx, 0.001 + n - sx)
   yBetaPostNoinfo = pdf(postBetaNoinfo, θ)
   plot!(θ, yBetaPostNoinfo, lw = 3, label = "a posteriori Beta no informativa")
end

