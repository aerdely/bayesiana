### Ejemplo: muestreador de Gibbs (distribución Normal)
### Autor: Dr. Arturo Erdely
### Versión: 2026-10-17

#=

    m.a X₁,…,Xₙ ~ Normal(μ,λ)

    E(X) = μ    V(X) = 1 / λ

=#


## Paquetes y código necesarios

using Distributions, Plots
include("student.jl")
include("02probestim.jl")



## Muestreador de Gibbs 

function gibbs(muestra::AbstractVector{<:Real}, μ₀::Real = 0.0, nsims::Integer = 100_000)
	n = length(muestra)
	n >= 2 || throw(ArgumentError("Se requiere tamaño de muestra ≥ 2"))
    	nsims > 0 || throw(ArgumentError("`nsims` debe ser positivo"))
	
	# ∑(xⱼ- μ)² = ∑(xⱼ- m)² + n(m - μ)² donde m = (∑xⱼ)/n

	mx = mean(muestra)
	S = sum(x -> abs2(x - mx), muestra)
	S > 0 || throw(ArgumentError("La muestra debe tener variabilidad"))

	μ = Vector{Float64}(undef, nsims)
	λ = Vector{Float64}(undef, nsims)
	m = Float64(μ₀)	

	for i in eachindex(μ)
		# distribuciones condicionales no informativas
		λ[i] = rand(Gamma(n/2, 2/(S + n*(mx - m)^2)))
		μ[i] = rand(Normal(mx, 1/sqrt(n*λ[i])))
		m = μ[i]
	end
	return (; μ, λ)
end

function levy_empirica(X::AbstractVector{<:Real},
                       Y::AbstractVector{<:Real};
                       tol::Float64 = 1e-10)

    isempty(X) && throw(ArgumentError("X está vacía"))
    isempty(Y) && throw(ArgumentError("Y está vacía"))
    tol > 0 || throw(ArgumentError("tol debe ser positiva"))

    x = sort(Float64.(X))
    y = sort(Float64.(Y))

    all(isfinite, x) && all(isfinite, y) ||
        throw(ArgumentError("Se requieren valores finitos"))

    n, m = length(x), length(y)

    function cumple(ε)
        for i in 1:n
            if i/n > searchsortedlast(y, x[i] + ε)/m + ε
                return false
            end
        end

        for j in 1:m
            if j/m > searchsortedlast(x, y[j] + ε)/n + ε
                return false
            end
        end

        return true
    end

    cumple(0.0) && return 0.0

    a, b = 0.0, 1.0

    while b - a > tol
        ε = (a + b)/2

        if cumple(ε)
            b = ε
        else
            a = ε
        end
    end

    return b
end

function diagnostico_levy(x::AbstractVector{<:Real};
                          bloque::Int = 10_000, gráfica::Bool = true)

    bloque > 0 || throw(ArgumentError("Bloque inválido"))
    length(x) % bloque == 0 || throw(ArgumentError("Longitud no divisible por bloque"))

    k = div(length(x), bloque)
    k >= 2 || throw(ArgumentError("Se requieren al menos 2 bloques"))

    F = [@view x[(i-1)*bloque+1:i*bloque] for i in 1:k]

    consecutivas = [levy_empirica(F[i], F[i+1]) for i in 1:k-1]

    contra_final = [levy_empirica(F[i], F[end]) for i in 1:k-1]

    if gráfica
	println("Diagnóstico de Lévy de la cadena")
	println("--------------------------------")
	println("Longitud de la cadena: ", length(x))
	println("Tamaño por bloque: ", bloque)
	println("Distancia máxima entre bloques:")
	println("consecutivos = ", maximum(consecutivas))
	println("contra final = ", maximum(contra_final))
	plot(consecutivas, ylabel = "distancia de Lévy", label = "consecutivas", lw = 2)
	title!("Distancia de Lévy entre bloques de tamaño $(bloque)")
	g = plot!(contra_final, label = "contra final", lw = 2)
	return (; consecutivas, contra_final, gráfica = g)
    else
	return (; consecutivas, contra_final)
    end
end




## Ejemplo

begin
	μ = -27.5
	λ = 14.2
	n = 100
	muestra = rand(Normal(μ, 1/sqrt(λ)), n)
	μ₀ = 0.0
	nsims = 100_000
	@time G = gibbs(muestra, μ₀, nsims)
	println("Estimaciones empíricas:")
	μ_emp = mean(muestra)
	λ_emp = 1 / var(muestra) 
	@show μ_emp
	@show λ_emp 
end;

[G.μ G.λ]



## Análisis de cadena marginal para μ

# parte inicial
begin
	pos_ini = 1    # del paso
	pos_fin = 100  # al paso
	μpos = G.μ[pos_ini:pos_fin]
	plot(μpos, label = "", xaxis = "n = $(pos_ini) al $(pos_fin)", yaxis = "μ")
	scatter!(μpos, label = "")
	hline!([μ], label = "teórica = $μ", lw = 2, color = :black)
	tray1 = hline!([μ_emp], label = "empírica = $(round(μ_emp, digits = 4))", lw = 2, color = :red)
end

# parte final
begin
	nselec = 100 # últimas al final de la cadena
	μpos = G.μ[(end - nselec + 1):end]
	plot(μpos, label = "", xaxis = "n (últimas $nselec)", yaxis = "μ")
	scatter!(μpos, label = "")
	hline!([μ], label = "teórica = $μ", lw = 2, color = :black)
	tray2 = hline!([μ_emp], label = "empírica = $(round(μ_emp, digits = 4))", lw = 2, color = :red)
end

# las dos anteriores en una misma gráfica
plot(tray1, tray2, layout = (2, 1)) 

# diagnóstico de Lévy
begin
	Dμ = diagnostico_levy(G.μ, bloque = 10_000)
	Dμ.gráfica
end

# parte final seleccionada
begin  
	nselec = 10_000 # últimas al final de la cadena
	μselec = G.μ[(end - nselec + 1):end]
	plot(μselec, label = "", xaxis = "n (últimas $nselec)", yaxis = "μ")
	hline!([μ], label = "teórica = $μ", lw = 2, color = :black)
	c1 = hline!([μ_emp], label = "empírica = $(round(μ_emp, digits = 4))", lw = 2, color = :red)
end

# densidad a posteriori no informativa: μ
begin
	nobs = length(muestra)
	media_obs = mean(muestra)
	var_obs = var(muestra, corrected = false)
	μ_post = rvStudent3(media_obs, sqrt(var_obs/nobs), nobs)
	μ_min, μ_max = quantile(G.μ, [0.001, 0.999])
	μ_valores = collect(range(μ_min, μ_max, length = 1_000))
	histogram(μselec, normalize = true, label = "Gibbs", 
                  xaxis = "μ", yaxis = "densidad a posteriori", color = :yellow)
	plot!(μ_valores, μ_post.pdf.(μ_valores), lw = 2, label = "Conjugada", color = :red)
	vline!([μ], label = "teórica = $μ", lw = 3, color = :blue)
	vline!([μ_emp], label = "empírica = $(round(μ_emp, digits = 4))", lw = 3, color = :violet)
end

# estimación puntual y por intervalo para μ
begin
	println("Estimación")
	println("----------")
	μ_est = mean(μselec)
	println("puntual: μ = ", μ_est)
	println("intervalo 95% = ", intervalo(μselec, 0.95).int)
end




## Análisis de cadena marginal para λ

# parte inicial
begin
	pos_ini = 1    # del paso
	pos_fin = 100  # al paso
	λpos = G.λ[pos_ini:pos_fin]
	plot(λpos, label = "", xaxis = "n = $(pos_ini) al $(pos_fin)", yaxis = "λ")
	scatter!(λpos, label = "")
	hline!([λ], label = "teórica = $λ", lw = 2, color = :black)
	tray3 = hline!([λ_emp], label = "empírica = $(round(λ_emp, digits = 4))", lw = 2, color = :red)
end

# parte final
begin
	nselec = 100 # últimas al final de la cadena
	λpos = G.λ[(end - nselec + 1):end]
	plot(λpos, label = "", xaxis = "n (últimas $nselec)", yaxis = "λ")
	scatter!(λpos, label = "")
	hline!([λ], label = "teórica = $λ", lw = 2, color = :black)
	tray4 = hline!([λ_emp], label = "empírica = $(round(λ_emp, digits = 4))", lw = 2, color = :red)
end

# las dos anteriores en una misma gráfica
plot(tray3, tray4, layout = (2, 1)) 

# diagnóstico de Lévy
begin
	Dλ = diagnostico_levy(G.λ, bloque = 10_000)
	Dλ.gráfica
end


# parte final seleccionada
begin  
	nselec = 10_000 # últimas al final de la cadena
	λselec = G.λ[(end - nselec + 1):end]
	plot(λselec, label = "", xaxis = "n (últimas $nselec)", yaxis = "λ")
	hline!([λ], label = "teórica = $λ", lw = 2, color = :black)
	c2 = hline!([λ_emp], label = "empírica = $(round(λ_emp, digits = 4))", lw = 2, color = :red)
end

# densidad a posteriori no informativa: λ
begin
	λ_post = Gamma(nobs / 2, 2 / (nobs * var_obs))
	λ_min, λ_max = quantile(G.λ, [0.001, 0.999])
	λ_valores = collect(range(λ_min, λ_max, length = 1_000))
	histogram(λselec, normalize = true, label = "Gibbs", 
                  xaxis = "λ", yaxis = "densidad a posteriori", color = :yellow)
	plot!(λ_valores, pdf(λ_post, λ_valores), lw = 2, label = "Conjugada", color = :red)
	vline!([λ], label = "teórica = $λ", lw = 3, color = :blue)
	vline!([λ_emp], label = "empírica = $(round(λ_emp, digits = 4))", lw = 3, color = :violet)
end

# estimación puntual y por intervalo para λ
begin
	println("Estimación")
	println("----------")
	λ_est = mean(λselec)
	println("puntual: λ = ", λ_est)
	println("intervalo 95% = ", intervalo(λselec, 0.95).int)
end



## Análisis de cadena conjunta para (μ,λ)

[G.μ G.λ]

# inicio de la cadena
begin
	n_ini = 100
	plot(G.μ[1:n_ini], G.λ[1:n_ini], xlabel = "μ", ylabel = "λ",
	     label = "", color = :gray, lw = 0.5)
	scatter!(G.μ[1:n_ini], G.λ[1:n_ini], label = "primeros $(n_ini) pasos", mc = :red)
	scatter!([G.μ[1]], [G.λ[1]], label = "n = 1", mc = :yellow)
end

# inicio y final de la cadena
begin
	n_fin = 100
	plot!(G.μ[(end - n_fin + 1):end], G.λ[(end - n_fin + 1):end], label = "", 
	      color = :black, lw = 0.5)
	scatter!(G.μ[(end - n_fin + 1):end], G.λ[(end - n_fin + 1):end], mc = :green,
	         label = "últimos $(n_fin) pasos")
end

# final de la cadena
begin
	n_fin = 10_000
	μ_fin = G.μ[(end - n_fin + 1):end]
	λ_fin = G.λ[(end - n_fin + 1):end]
	scatter(μ_fin, λ_fin, xlabel = "μ", ylabel = "λ", label = "", 
                mc = :black, ms = 1.0, title = "últimas $(n_fin) simulaciones")
	scatter!([μ], [λ], mc = :cyan, label = "valor teórico")
	scatter!([μ_est], [λ_est], mc = :red, label = "estimación puntual")
end

# densidad a posteriori conjunta de (μ,λ), familia conjugada Normal-Gamma
begin
	p(μ, λ) = pdf(λ_post, λ) * pdf(Normal(media_obs, 1 / sqrt(nobs*λ)), μ)
	μ_min, μ_max = quantile(G.μ, [0.001, 0.999])
	μ_valores = collect(range(μ_min, μ_max, length = 1_000))
	λ_min, λ_max = extrema(G.λ)
	λ_valores = collect(range(λ_min, λ_max, length = 1_000))
	z = [p(μ, λ) for λ in λ_valores, μ in μ_valores]
	contour(μ_valores, λ_valores, z, xlabel = "μ", ylabel = "λ", fill = true)
end

# agregar simulaciones de Gibbs
begin
	scatter!(μ_fin, λ_fin, label = "Gibbs", ms = 0.7, mc = :green, markerstrokecolor = :green)
	scatter!([μ], [λ], mc = :cyan, label = "valor teórico")
	scatter!([μ_est], [λ_est], mc = :red, label = "estimación puntual")
end



## Coeficiente de variación: cv = σ/|μ| = 1/(|μ|λ^(1/2))

# densidad a posteriori no informativa del cv
begin
	cv = @. 1 / (abs(μ_fin) * sqrt(λ_fin))
	histogram(cv, normalize = true, title = "Coeficiente de variación a posteriori", label = "Gibbs")
	xaxis!("cv"); yaxis!("densidad a posteriori")
	println("Estimaciones coeficiente de variación")
	println("-------------------------------------")
	cv_teo = 1 / (abs(μ) * sqrt(λ))
	cv_emp = 1 / (abs(media_obs) * sqrt(λ_emp))
	cv_est = median(cv)
	println("teórica:       ", cv_teo)
	println("empírica:      ", cv_emp)
	println("estimada:      ", cv_est)
	cv_int = intervalo(cv, 0.95).int
	println("intervalo 95%: ", cv_int)
	scatter!([cv_teo], [0.0], label = "teórico", ms = 5, mc = :red)
	scatter!([cv_emp], [0.0], label = "empírico", ms = 5, mc = :pink)
	scatter!([cv_est], [0.0], label = "estimado", ms = 5, mc = :yellow)
	vline!(cv_int, label = "intervalo 95%", color = :green, lw = 2)
end



## Distribución predictiva a posteriori

# vía análisis conjugado
X = rvStudent3(media_obs, sqrt((nobs + 1)*var_obs/nobs), nobs); 

# vía estimación de esperanza a posteriori E(f(x|μ,λ))
function pp(x::Real, μs::AbstractVector, λs::AbstractVector)
    length(μs) == length(λs) || throw(DimensionMismatch("Longitudes distintas"))

    prom = 0.0

    for i in eachindex(μs, λs)
        prom += pdf(Normal(μs[i], 1/sqrt(λs[i])), x)
    end

    return prom / length(μs)
end

# simulación 
function simpp(μs::AbstractVector, λs::AbstractVector)
    length(μs) == length(λs) ||
        throw(DimensionMismatch("Longitudes distintas"))

    return [μs[i] + randn()/sqrt(λs[i])
            for i in eachindex(μs, λs)]
end

begin # ejecutar varias veces
	xsim = simpp(μselec, λselec)
	xmin, xmax = extrema(xsim)
	xval = collect(range(xmin, xmax, length = 1_000))
	histogram(xsim, normalize = true, label = "simulación Gibbs",
        	  color = :yellow, xlabel = "x", ylabel = "densidad predictiva a posteriori")
end
plot!(xval, pp.(xval, Ref(μselec), Ref(λselec)), lw = 4, color = :violet, label = "promedio Gibbs")
plot!(xval, X.pdf.(xval), lw = 2, color = :red, label = "conjugada")
plot!(xval, pdf(Normal(μ, 1/sqrt(λ)), xval), lw = 2, color = :blue, label = "teórica")


