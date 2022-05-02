"""
    Canonical Ensemble Projections

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number
    expβϵ -> exponentiated spectrum, i.e., exp(-βϵ)
    expiφ -> discrete Fourier frequencies
"""

function pf_projection(
    Ns::Int64, N::Int64, expβϵ::Vector{T}, 
    expiφ::Vector{ComplexF64}, returnFull::Bool
) where {T<:FloatType}

    expβμ = fermilevel(expβϵ, N)

    η = [prod(1 .+ expiφ[m] / expβμ * expβϵ) for m = 1 : Ns + 1]
    Z = sum(conj(expiφ) .^ N .* η) / (Ns + 1) * expβμ ^ N

    returnFull || return Z

    return expβμ, η, Z

end

function occ_projection(
    Ns::Int64, N::Int64, expβϵ::Vector{T}, 
    expiφ::Vector{ComplexF64}
) where {T<:FloatType}
    expβμ, η, Z = pf_projection(Ns, N, expβϵ, expiφ, true)
    ñ = [expiφ[m] / expβμ * expβϵ[i] / (1 + expiφ[m] / expβμ * expβϵ[i]) for m = 1 : Ns + 1, i = 1 : Ns]
    n = [sum(conj(expiφ) .^ N .* ñ[:, i] .* η) * expβμ ^ N / (Ns + 1) / Z for i = 1 : Ns]

    return n

end

function doubleocc_projection()
end