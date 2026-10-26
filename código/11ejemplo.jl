### Ejemplo: muestreador de Gibbs (regresión simple)
### Autor: Dr. Arturo Erdely
### Versión: 2026-10-19

#=

    Yₓ = α + βx + ε,  ε ~ Uniforme(-σ, σ)

         α, β ∈ ℜ    σ > 0

    ⇒ Yₓ ~ Uniforme(α + βx - σ, α + βx + σ)

    observs indeps: D = {(xᵢ,yₓᵢ): i = 1,...,n}

=#


## Paquetes y código necesarios

using Distributions, Plots, StatsBase
include("02probestim.jl")



## Funciones auxiliares

function R(α::Real, β::Real, muestra::AbstractMatrix{<:Real})
	# `muestra` es una matriz de observaciones con valores (xᵢ,yₓᵢ) por fila
	x = muestra[:, 1]
	y = muestra[:, 2]
	return maximum(abs.(y .- α .- (β .* x)))	
end

function IS1(β::Real, σ::Real, muestra::AbstractMatrix{<:Real})
	# `muestra` es una matriz de observaciones con valores (xᵢ,yₓᵢ) por fila
	x = muestra[:, 1]
	y = muestra[:, 2]
	I1 = maximum(y .- (β .* x) .- σ)
	S1 = minimum(y .- (β .* x) .+ σ)
	return(I1, S1)
end

function IS2(α::Real, σ::Real, muestra::AbstractMatrix{<:Real})
	# `muestra` es una matriz de observaciones con valores (xᵢ,yₓᵢ) por fila
	idx = findall(muestra[:, 1] .≠ 0.0)
	nx = length(idx)
	x = muestra[idx, 1]
	y = muestra[idx, 2]
	τ = Vector{Float64}(undef, nx)
	ω = Vector{Float64}(undef, nx)
	for i ∈ eachindex(x)
		τ[i], ω[i] = extrema([(y[i] - α - σ)/x[i], (y[i] - α + σ)/x[i]])
	end
	I2 = maximum(τ)
	S2 = minimum(ω)	
	return(I2, S2)
end


## Muestreador de Gibbs 

function gibbs(muestra::AbstractMatrix{<:Real}, α₀::Real = 0.0, β₀::Real = 0.0, nsims::Integer = 100_000)
	# `muestra` es una matriz de observaciones con valores (xᵢ,yₓᵢ) por fila
	# utiliza las funciones: ´R´, ´IS1´, ´IS2´
	n = size(muestra, 1)
	nsims > 0 || throw(ArgumentError("`nsims` debe ser positivo"))

	α = Vector{Float64}(undef, nsims + 1)
	β = Vector{Float64}(undef, nsims + 1)
	σ = Vector{Float64}(undef, nsims + 1)

	α[1] = α₀
	β[1] = β₀
	σ[1] = 0.0

	for i ∈ 2:(nsims + 1)
		σ[i] = rand(Pareto(n, R(α[i-1], β[i-1], muestra)))
		IS = IS1(β[i-1], σ[i], muestra)
		α[i] = rand(Uniform(IS[1], IS[2]))
		IS = IS2(α[i], σ[i], muestra)
		β[i] = rand(Uniform(IS[1], IS[2]))
	end
	return(α = α[2:end], β = β[2:end], σ = σ[2:end])
end



## Ejemplo

function ejemplo(α::Real, β::Real, σ::Real, x::AbstractVector{<:Real})
	# simula observaciones (x, Yₓ)
	n = length(x)
	Y = Vector{Float64}(undef, n)
	for i ∈ eachindex(x)
		Y[i] = rand(Uniform(α + β*x[i] - σ, α + β*x[i] + σ))
	end
	return [x Y]
end

begin
	α = -3.5
	β = 2.1
	σ = 7.6
	x = collect(range(-10, 10, length = 301))
	xY = ejemplo(α, β, σ, x)
	scatter(xY[:, 1], xY[:, 2], legend = false, xlabel = "x", ylabel = "Yₓ")
end

begin
	@time G = gibbs(xY, 0.0, 0.0, 100_000)
	[G.α G.β G.σ]
end




## Análisis de cadena marginal para α

# parte inicial
begin
	pos_ini = 1    # del paso
	pos_fin = 100  # al paso
	αpos = G.α[pos_ini:pos_fin]
	plot(αpos, label = "", xaxis = "n = $(pos_ini) al $(pos_fin)", yaxis = "α")
	scatter!(αpos, label = "")
	tray1 = hline!([α], label = "valor teórico = $α", lw = 2, color = :black)
end

# parte final
begin
	nselec = 100 # últimas al final de la cadena
	αpos = G.α[(end - nselec + 1):end]
	plot(αpos, label = "", xaxis = "n (últimas $nselec)", yaxis = "α")
	scatter!(αpos, label = "")
	tray2 = hline!([α], label = "valor teórico = $α", lw = 2, color = :black)
end

# las dos anteriores en una misma gráfica
plot(tray1, tray2, layout = (2, 1)) 

# parte final seleccionada
begin  
	nselec = 10_000 # últimas al final de la cadena
	αselec = G.α[(end - nselec + 1):end]
	plot(αselec, label = "", xaxis = "n (últimas $nselec)", yaxis = "α")
	c1 = hline!([α], label = "valor teórico = $α", lw = 2, color = :black)
end

# densidad a posteriori no informativa: α
begin
	histogram(αselec, normalize = true, label = "Gibbs", 
                  xaxis = "α", yaxis = "densidad a posteriori", color = :yellow)
	vline!([α], label = "valor teórico = $α", lw = 3, color = :blue)
end

# estimación puntual y por intervalo para α
begin
	println("Estimación")
	println("----------")
	α_est = median(αselec)
	println("puntual: α = ", α_est)
	println("intervalo 95% = ", intervalo(αselec, 0.95).int)
end



## Análisis de cadena marginal para β

# parte inicial
begin
	pos_ini = 1    # del paso
	pos_fin = 100  # al paso
	βpos = G.β[pos_ini:pos_fin]
	plot(βpos, label = "", xaxis = "n = $(pos_ini) al $(pos_fin)", yaxis = "β")
	scatter!(βpos, label = "")
	tray1 = hline!([β], label = "valor teórico = $β", lw = 2, color = :black)
end

# parte final
begin
	nselec = 100 # últimas al final de la cadena
	βpos = G.β[(end - nselec + 1):end]
	plot(βpos, label = "", xaxis = "n (últimas $nselec)", yaxis = "β")
	scatter!(βpos, label = "")
	tray2 = hline!([β], label = "valor teórico = $β", lw = 2, color = :black)
end

# las dos anteriores en una misma gráfica
plot(tray1, tray2, layout = (2, 1)) 

# parte final seleccionada
begin  
	nselec = 10_000 # últimas al final de la cadena
	βselec = G.β[(end - nselec + 1):end]
	plot(βselec, label = "", xaxis = "n (últimas $nselec)", yaxis = "β")
	c1 = hline!([β], label = "valor teórico = $β", lw = 2, color = :black)
end

# densidad a posteriori no informativa: β
begin
	histogram(βselec, normalize = true, label = "Gibbs", 
                  xaxis = "β", yaxis = "densidad a posteriori", color = :yellow)
	vline!([β], label = "valor teórico = $β", lw = 3, color = :blue)
end

# estimación puntual y por intervalo para β
begin
	println("Estimación")
	println("----------")
	β_est = median(βselec)
	println("puntual: β = ", β_est)
	println("intervalo 95% = ", intervalo(βselec, 0.95).int)
end



## Análisis de cadena marginal para σ

# parte inicial
begin
	pos_ini = 1    # del paso
	pos_fin = 100  # al paso
	σpos = G.σ[pos_ini:pos_fin]
	plot(σpos, label = "", xaxis = "n = $(pos_ini) al $(pos_fin)", yaxis = "σ")
	scatter!(σpos, label = "")
	tray1 = hline!([σ], label = "valor teórico = $σ", lw = 2, color = :black)
end

# parte final
begin
	nselec = 100 # últimas al final de la cadena
	σpos = G.σ[(end - nselec + 1):end]
	plot(σpos, label = "", xaxis = "n (últimas $nselec)", yaxis = "σ")
	scatter!(σpos, label = "")
	tray2 = hline!([σ], label = "valor teórico = $σ", lw = 2, color = :black)
end

# las dos anteriores en una misma gráfica
plot(tray1, tray2, layout = (2, 1)) 

# parte final seleccionada
begin  
	nselec = 10_000 # últimas al final de la cadena
	σselec = G.σ[(end - nselec + 1):end]
	plot(σselec, label = "", xaxis = "n (últimas $nselec)", yaxis = "σ")
	c1 = hline!([σ], label = "valor teórico = $σ", lw = 2, color = :black)
end

# densidad a posteriori no informativa: σ
begin
	histogram(σselec, normalize = true, label = "Gibbs", 
                  xaxis = "σ", yaxis = "densidad a posteriori", color = :yellow)
	vline!([σ], label = "valor teórico = $σ", lw = 3, color = :blue)
end

# estimación puntual y por intervalo para σ
begin
	println("Estimación")
	println("----------")
	σ_est = median(σselec)
	println("puntual: σ = ", σ_est)
	println("intervalo 95% = ", intervalo(σselec, 0.95).int)
end




### Análisis de cadenas conjuntas: (α, β), (α, σ), (β, σ) y (α, β, σ)

## (α, β)

# inicio de la cadena (α, β)
begin
	n_ini = 100
	plot(G.α[1:n_ini], G.β[1:n_ini], xlabel = "α", ylabel = "β",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.α[1:n_ini], G.β[1:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.α[1]], [G.β[1]], label = "n = 1", mc = :yellow)
end

# inicio y final de la cadena
begin
	n_fin = 100
	plot!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# inicio de la cadena (α, β) a partir de n = nk > 1
begin
	nk = 8
	n_ini = 100
	plot(G.α[nk:n_ini], G.β[nk:n_ini], xlabel = "α", ylabel = "β",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.α[nk:n_ini], G.β[nk:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.α[nk]], [G.β[nk]], label = "n = $(nk)", mc = :yellow)
end

# inicio en nk y final de la cadena
begin
	n_fin = 100
	plot!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# final de la cadena
begin
	n_fin = 10_000
	α_fin = G.α[(end - n_fin + 1):end]
	β_fin = G.β[(end - n_fin + 1):end]
	corsp_αβ = corspearman(α_fin, β_fin)
	scatter(α_fin, β_fin, xlabel = "α", ylabel = "β", label = "", 
                mc = :black, ms = 1.0, title = "últimas $(n_fin) simulaciones | Spearman = $(round(corsp_αβ, digits = 4))")
	scatter!([α], [β], mc = :cyan, label = "valor teórico")
	scatter!([α_est], [β_est], mc = :red, label = "estimación puntual")
end


## (α, σ)

# inicio de la cadena (α, σ)
begin
	n_ini = 100
	plot(G.α[1:n_ini], G.σ[1:n_ini], xlabel = "α", ylabel = "σ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.α[1:n_ini], G.σ[1:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.α[1]], [G.σ[1]], label = "n = 1", mc = :yellow)
end

# inicio y final de la cadena
begin
	n_fin = 100
	plot!(G.α[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.α[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# inicio de la cadena (α, σ) a partir de n = nk > 1
begin
	nk = 8
	n_ini = 100
	plot(G.α[nk:n_ini], G.σ[nk:n_ini], xlabel = "α", ylabel = "σ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.α[nk:n_ini], G.σ[nk:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.α[nk]], [G.σ[nk]], label = "n = $(nk)", mc = :yellow)
end

# inicio en nk y final de la cadena
begin
	n_fin = 100
	plot!(G.α[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.α[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# final de la cadena
begin
	n_fin = 10_000
	α_fin = G.α[(end - n_fin + 1):end]
	σ_fin = G.σ[(end - n_fin + 1):end]
	corsp_ασ = corspearman(α_fin, σ_fin)
	scatter(α_fin, σ_fin, xlabel = "α", ylabel = "σ", label = "", 
                mc = :black, ms = 1.0, title = "últimas $(n_fin) simulaciones | Spearman = $(round(corsp_ασ, digits = 4))")
	scatter!([α], [σ], mc = :cyan, label = "valor teórico")
	scatter!([α_est], [σ_est], mc = :red, label = "estimación puntual")
end


## (β, σ)

# inicio de la cadena (β, σ)
begin
	n_ini = 100
	plot(G.β[1:n_ini], G.σ[1:n_ini], xlabel = "β", ylabel = "σ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.β[1:n_ini], G.σ[1:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.β[1]], [G.σ[1]], label = "n = 1", mc = :yellow)
end

# inicio y final de la cadena
begin
	n_fin = 100
	plot!(G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# inicio de la cadena (β, σ) a partir de n = nk > 1
begin
	nk = 8
	n_ini = 100
	plot(G.β[nk:n_ini], G.σ[nk:n_ini], xlabel = "β", ylabel = "σ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.β[nk:n_ini], G.σ[nk:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.β[nk]], [G.σ[nk]], label = "n = $(nk)", mc = :yellow)
end

# inicio en nk y final de la cadena
begin
	n_fin = 100
	plot!(G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# final de la cadena
begin
	n_fin = 10_000
	β_fin = G.β[(end - n_fin + 1):end]
	σ_fin = G.σ[(end - n_fin + 1):end]
	corsp_βσ = corspearman(β_fin, σ_fin)
	scatter(β_fin, σ_fin, xlabel = "β", ylabel = "σ", label = "", 
                mc = :black, ms = 1.0, title = "últimas $(n_fin) simulaciones | Spearman = $(round(corsp_βσ, digits = 4))")
	scatter!([β], [σ], mc = :cyan, label = "valor teórico")
	scatter!([β_est], [σ_est], mc = :red, label = "estimación puntual")
end


## (α, β, σ)

# inicio de la cadena (α, β, σ)
begin
	n_ini = 100
	plot(G.α[1:n_ini], G.β[1:n_ini], G.σ[1:n_ini], xlabel = "α", ylabel = "β", zlabel = "σ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.α[1:n_ini], G.β[1:n_ini], G.σ[1:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.α[1]], [G.β[1]], [G.σ[1]], label = "n = 1", mc = :yellow)
end

# inicio y final de la cadena
begin
	n_fin = 100
	plot!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# inicio de la cadena (α, β, σ) a partir de n = nk > 1
begin
	nk = 8
	n_ini = 100
	plot(G.α[nk:n_ini], G.β[nk:n_ini], G.σ[nk:n_ini], xlabel = "α", ylabel = "β", zlabel = "σ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.α[nk:n_ini], G.β[nk:n_ini], G.σ[nk:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.α[nk]], [G.β[nk]], [G.σ[nk]], label = "n = $(nk)", mc = :yellow)
end

# inicio en nk y final de la cadena
begin
	n_fin = 100
	plot!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.α[(end - n_fin + 1):end], G.β[(end - n_fin + 1):end], G.σ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# final de la cadena
begin
	n_fin = 3_000
	α_fin = G.α[(end - n_fin + 1):end]
	β_fin = G.β[(end - n_fin + 1):end]
	σ_fin = G.σ[(end - n_fin + 1):end]
	scatter(α_fin, β_fin, σ_fin, xlabel = "α", ylabel = "β", zlabel = "σ", label = "", 
                mc = :black, ms = 1.0, title = "últimas $(n_fin) simulaciones")
	scatter!([α], [β], [σ], mc = :cyan, label = "valor teórico")
	scatter!([α_est], [β_est], [σ_est], mc = :red, label = "estimación puntual")
end


