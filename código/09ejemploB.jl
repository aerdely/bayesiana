### Ejemplo muestreador de Gibbs
### Autor: Dr. Arturo Erdely
### Versión: 2026-10-11

#=
    m.a  X₁,…,Xₙ ~ Binomial(r,θ)
    r ∈ {1,2,…}   0 < θ < 1

         (r,θ) = ?
=#


## Paquetes y código necesarios

using Distributions, Plots, QuadGK
include("02probestim.jl")
include("03discreta.jl")



## Distribución a priori para 'r'

function π₁(r::Int, α₁::Float64)
	return r^(-α₁) - (r+1)^(-α₁)
end

# probando
f(x) = π₁(Int(floor(x)), 1.0)
quadgk(f, 1, Inf)


## Distribución a posteriori condicional: r|θ,xx

function logL₁π₁(r::Int, θ::Float64, α₁::Float64, xx::Vector{Int})
	mx = maximum(xx)
	if r < mx
		return -Inf
	else
		n = length(xx)
		sx = sum(xx)
		A = 0.0
		for i in 1:n
			for k in 0:(xx[i]-1)
				A += log(r-k) - log(xx[i]-k)
			end
		end
		return sx*log(θ) + (n*r - sx)*log(1-θ) + log(π₁(r,α₁)) + A
	end
end

function L₁π₁(r::Int, θ::Float64, α₁::Float64, xx::Vector{Int})
	mx = maximum(xx)
	if r < mx
		return 0.0
	else
		return exp(logL₁π₁(r, θ, α₁, xx))
	end
end

function C₁(θ::Float64, α₁::Float64, xx::Vector{Int}; err = false)
	mx = maximum(xx)
	f(x) = L₁π₁(Int(floor(x)), θ, α₁, xx)
	Q = quadgk(f, mx, Inf)
	if err
		println("Error máx: ", Q[2])
	end
	return Q[1]
end

xx = rand(Binomial(10, 0.38), 20)

C₁(0.38, 1.0, xx, err = true)

function p₁(r::Int, θ::Float64, α₁::Float64, xx::Vector{Int})
	# a posteriori: r|θ,xx
	return L₁π₁(r, θ, α₁, xx) / C₁(θ, α₁, xx)
end

# probando
f(x) = p₁(Int(floor(x)), 0.5, 1.0, xx)
quadgk(f, 1, Inf)

function sim_p₁(θ::Float64, α₁::Float64, xx::Vector{Int})
	p(r) = p₁(r, θ, α₁, xx)
	return simDiscreta(p, 1)[1]
end



## Distribución a posteriori condicional: θ|r,xx

function sim_p₂(r::Int, α₂::Float64, β₂::Float64, xx::Vector{Int})
	n = length(xx)
	sx = sum(xx)
	return rand(Beta(α₂ + sx, β₂ + n*r - sx))
end



## Muestreador de Gibbs 

function gibbs(θ₀::Float64, nsims::Int, α₁::Float64, α₂::Float64, β₂::Float64, xx::Vector{Int})
	r = zeros(Int, nsims + 1)
	θ = zeros(nsims + 1)
	θ[1] = θ₀
	for i in 2:(nsims+1)
		r[i] = sim_p₁(θ[i-1], α₁, xx)
		θ[i] = sim_p₂(r[i], α₂, β₂, xx)
	end
	return (r = r[2:end], θ = θ[2:end])
end

function ejemplo(r::Int, θ::Float64, n::Int, θ₀::Float64, nsims::Int, α₁::Float64, α₂::Float64, β₂::Float64)
	xx = rand(Binomial(r, θ), n)
	return gibbs(θ₀, nsims, α₁, α₂, β₂, xx)
end

G = ejemplo(10, 0.38, 20, 0.5, 1_000, 1.0, 0.5, 0.5);
[G.r G.θ]

