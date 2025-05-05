### Regresión lineal simple bajo normalidad
### Autor: Dr. Arturo Erdely
### Versión: 2025-04-27


# Cargar paquetes necesarios:

using Random # no requiere instalación previa
using Distributions, Plots, LaTeXStrings # previamente instalados


# Simular muestra condicionada a partir del modelo Yx = α + βx + ε
begin
    α, β = 2, -5
    σ2 = 100
    x = collect(range(-5, 7, length = 300))
    n = length(x)
    ε = Normal(0, √σ2)
    Random.seed!(4321)
    y = α .+ β.*x .+ rand(ε, n)
    scatter(x, y, label = "muestra observada", mc = :yellow, xlabel = L"x", ylabel = L"Y_{\!\!\! x}")
    plot!(x, α .+ β.*x, label = "recta teórica", color = :blue)
end


# Estimación puntual de α + βx por mínimos cuadrados ordinarios
begin
    sx = sum(x)
    sx2 = sum(x .^ 2)
    sy = sum(y)
    sxy = sum(x .* y)
    dx = n*sx2 - sx^2
    a = (sx2*sy - sx*sxy) / dx 
    b = (n*sxy - sx*sy) / dx 
    g(x) = a + b*x 
    plot!(x, g.(x), lw = 2, color = :red, label = "recta estimada")
end


# Estimación de σ²
begin
    e = y - g.(x) # residuos
    s2 = var(e)
    q975 = quantile(Normal(0,1), 0.975)
    plot!(x, g.(x) .+ q975*√s2, color = :violet, label = "")
    plot!(x, g.(x) .- q975*√s2, color = :violet, label = "probabilidad 95%")
end



# Método ABC
@time begin
    m = 1_000_000 # simulaciones
    k = 100 # cantidad a seleccionar
    sumstat(x,y) = [sum(y), sum(x .* y), sum(y .^ 2)] # summary statistic
    obstat = sumstat(x, y) # valor observado del summary statistic
    δ(u, v) = √sum((u - v) .^2) # distancia euclidiana
    dist = zeros(m) # inicializar vector de distancias
    aa = rand(Uniform(a - 2*abs(a), a + 2*abs(a)), m) # simular a priori 
    bb = rand(Uniform(b - 2*abs(b), b + 2*abs(b)), m) # simular a priori 
    ss = rand(Uniform(0, 2*s2), m) # simular a priori 
    for i ∈ 1:m 
        ysim = aa[i] .+ bb[i].*x .+ rand(Normal(0, √ss[i]), n)
        dist[i] = δ(obstat, sumstat(x, ysim)) # distancia observado vs simulado
    end
    iSelec = sortperm(dist)[1:k] # simulaciones a seleccionar
    αpost = aa[iSelec] # simulaciones a posteriori
    βpost = bb[iSelec] # simulaciones a posteriori
    σ2post = ss[iSelec] # simulaciones a posteriori
end;


# Gráficas de parámetros a posteriori

begin
    αest = median(αpost)
    βest = median(βpost)
    σ2est = median(σ2post)
    histogram(αpost, normalize = true, color = :yellow,
              label = "simulaciones a posteriori",
              xlabel = "α", ylabel = "densidad a posteriori"
    )
    vline!([α], label = "valor teórico", color = :blue, lw = 3)
    αgraf = vline!([αest], label = "mediana a posteriori", color = :red, lw = 3)
end

begin
    histogram(βpost, normalize = true, color = :lightgreen,
              label = "simulaciones a posteriori",
              xlabel = "β", ylabel = "densidad a posteriori"
    )
    vline!([β], label = "valor teórico", color = :blue, lw = 3)
    βgraf = vline!([βest], label = "mediana a posteriori", color = :red, lw = 3)
end

begin
    histogram(σ2post, normalize = true, color = :pink,
              label = "simulaciones a posteriori",
              xlabel = L"\sigma^2", ylabel = "densidad a posteriori"
    )
    vline!([σ2], label = "valor teórico", color = :blue, lw = 3)
    σ2graf = vline!([σ2est], label = "mediana a posteriori", color = :red, lw = 3)
end

begin
    scatter(αpost, βpost, label = "simulaciones a posteriori", xlabel = "α", ylabel = "β", mc = :yellow)
    scatter!([α], [β], label = "valor teórico", mc = :blue, ms = 5)
    plot!([minimum(αpost), α], [β, β], label = "", color = :blue)
    plot!([α, α], [minimum(βpost), β], label = "", color = :blue)
    scatter!([αest], [βest], label = "mediana a posteriori", mc = :red, ms = 5)
    plot!([minimum(αpost), αest], [βest, βest], label = "", color = :red)
    αβgraf = plot!([αest, αest], [minimum(βpost), βest], label = "", color = :red)
end 

plot(βgraf, αβgraf, σ2graf, αgraf, layout = (2,2), size = (1000,800))
