### Ejemplo: muestreador de Gibbs con diagnóstico
### Autor: Dr. Arturo Erdely
### Versión: 2026-10-25

#=

    Yₓ = α + βx + ε,  ε ~ Uniforme(-σ, σ)

         α, β ∈ ℜ    σ > 0

    ⇒ Yₓ ~ Uniforme(α + βx - σ, α + βx + σ)

    observs indeps: D = {(xᵢ,yₓᵢ): i = 1,...,n}

=#


## Paquetes y código necesarios

using Distributions, Plots
include("levy.jl")


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

function diagnostico(cadena, tamsub::Int)
	n = length(cadena)
	k = tamsub
	id1 = collect(1:k)
	id2 = collect((n-2k+1):(n-k))
	id3 = collect((n-k+1):n)
	diag1 = levy_empirica(cadena[id1], cadena[id3])
	diag2 = levy_empirica(cadena[id2], cadena[id3])
	println("Tamaño subcadenas = ", k)
	println("p-valor Ho: bloques misma distribución")
	println("--------------------------------------")
	println("  primero vs último = ", diag1.p_boot)
	println("penúltimo vs último = ", diag2.p_boot)
	println("--------------------------------------")
	return (pini = diag1.p_boot, pfin = diag2.p_boot, 
                nsub = k, dini = diag1, dfin = diag2)
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
	ng = 100_000 # probar valores mayores de ser necesario
	@time MG = gibbs(xY, 0.0, 0.0, ng) 
	[MG.α MG.β MG.σ]
end

# Diagnósticos intracadenas
begin
	println("=== α ===")
	Dα = diagnostico(MG.α, length(x))
	println("=== β ===")
	Dβ = diagnostico(MG.β, length(x))
	println("=== σ ===")
	Dσ = diagnostico(MG.σ, length(x))	
end;
