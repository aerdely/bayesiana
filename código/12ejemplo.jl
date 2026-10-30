### Ejemplo: Distancia de Lévy
### Autor: Dr. Arturo Erdely
### Fecha: 2026-10-24


using Distributions, Plots
include("levy.jl")


## X ~ Normal(0,1), Y ~ Student(1)

begin
	x₀ = 1
	X = Normal(0,1)
	Y = TDist(1)
	F(x) = cdf(X, x)
	G(y) = cdf(Y, y)
	xx = collect(range(-5, 5, length = 1_000))
	plot(xx, F.(xx), lw = 2, label = "F", xlabel = "x", ylabel = "P(X ≤ x)")
	hline!([0], lw = 0.5, color = :black, label = "")
	hline!([1], lw = 0.5, color = :black, label = "")
	plot!(xx, G.(xx), lw = 2, label = "G")
	vline!([x₀], label = "x₀ = $(x₀)")
end

begin
	εₓ = levy_puntual(x₀, F, G)
	@show εₓ
	cotasup = G(x₀ + εₓ) + εₓ
	cotainf = G(x₀ - εₓ) - εₓ
	hline!([cotasup], color = :red, label = "sup = $(round(cotasup, digits = 4))")
	hline!([cotainf], color = :red, label = " inf = $(round(cotainf, digits = 4))")
end

begin
	εε = [levy_puntual(x, F, G) for x in xx]
	ε = findmax(εε)
	println("Distancia de Levy = ", ε[1])
	println("Máxima distancia puntual aprox se alcanza, al menos, en x = ", xx[ε[2]])
	plot(xx, εε, xlabel = "x", ylabel = "εₓ", label = "", legend = :right)
	vline!(xx[[ε[2]]], label = "x = $(round(xx[ε[2]], digits = 4))")
	hline!([ε[1]], label = "dL(F,G) = $(round(ε[1], digits = 4))")
end


## Distribuciones empíricas

# Ejecutar varias veces manteniendo fijos tamaños de muestra
# y luego probar con distintos tamaños de muestra

begin 
	nx = 120 # tamaño de muestra X
	ny = 90  # tamaño de muestra Y 
	xobs = rand(X, nx)
	yobs = rand(Y, ny)
	@time Levy = levy_empirica(xobs, yobs)
	println()
	println("Distancia de Lévy empírica")
	println("==========================")
	println("observada = ", Levy.dist)
	println()
	println("Bajo Ho: F = G")
	println("--------------")
	println("   dLo = ", Levy.Ho)
	println("   cuantiles 0.95 y 0.99 = ", (Levy.q95, Levy.q99))
	println("   p-valor = ", Levy.p_boot)
end
