"""
    Canonical Ensemble Recursions

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number (number of recursions)
    expβϵ -> exponentiated spectrum, i.e., exp(-βϵ)
"""
function pf_recursion(
    Ns::Int64, N::Int64, expβϵ::Vector{T}; returnFull::Bool = false, useDouble::Bool = true
) where {T<:FloatType}
    """
    Recursive calculation of the partition function
    """
    N == 0 && return convert(T, 0.0)
    N == 1 && return log(complex(sum(expβϵ)))
    N == Ns && return sum(log.(complex(expβϵ)))
    
    # rescale spectrum
    expβμ = fermilevel(expβϵ, N)
    expβϵμ = expβϵ / expβμ
    # map to higher precision
    useDouble && (expβϵμ = ComplexDF64.(expβϵμ))
    
    P = poissbino(Ns, expβϵμ, false)
    # unmap to Float64
    P = convert.(ComplexF64, P)
    if N > Ns / 2
        logZ = log(P[N + 1]) - log(P[end]) - (Ns - N)*log(expβμ) + sum(log.(complex(expβϵ)))
    else
        # non-logarithmic version: Z = P[N + 1] / P[2] * (expβμ) ^ (N - 1) * Z₁
        # use P[2] instead of P[1] as P[2] is more stable numerically
        logZ = log(P[N + 1]) - log(P[2]) + (N - 1)*log(expβμ) + log(complex(sum(expβϵ)))
    end

    returnFull || return logZ
    return expβϵμ, P, logZ
end

function occ_recursion(
    Ns::Int64, N::Int64, expβϵ::Vector{T}
) where {T<:FloatType}
    """
    Recursive calculation of the occupation number (in the momentum space)
    """
    N == 0 && return zeros(T, Ns)
    N == Ns && return ones(T, Ns)
    n₁ = expβϵ / sum(expβϵ)
    N == 1 && return n₁

    expβϵμ, P, logZ = pf_recursion(Ns, N, expβϵ, returnFull=true)
    # num of energy levels below the Fermi level
    # use this formula to ensure complex conjugate pairs are in the same section
    N_below = sum(abs.(expβϵμ) .> 1)

    # separate recursions for occupancies of energy levels above/below the Fermi level
    n_above = occ_recursion_rescaled(Ns, N, expβϵμ[1 : Ns - N_below], P, n₁, false)
    n_below = occ_recursion_rescaled(Ns, N, expβϵμ[Ns - N_below + 1 : Ns], P, n₁, true)
    # then concatenate
    n = vcat(n_above, n_below)
    # unmap to Float64
    n = convert.(ComplexF64, n)
    return n
end

function occ_recursion_rescaled(
    Ns::Int64, N::Int64,
    expβϵμ::Vector{Te}, P::Vector{Tp}, n₁::Vector{Tn},
    isReverse::Bool
) where {Te<:FloatType, Tp<:FloatType, Tn<:FloatType}
    """
    Level occupancy recursion using the rescaled spectrum
    """
    Ñs = length(expβϵμ)
    if !isReverse
        n = zeros(Tn, N + 1, Ñs)
        # use n₁ as a correction
        n[2, :] = n₁[1 : Ñs]
        for i = 2 : N
            n[i + 1, :] = (P[i] / P[i + 1]) * expβϵμ .* (1 .- n[i, :])
            # Truncate the values that are smaller than 10^-10
            n[i + 1, :] = n[i + 1, :] .* (abs.(real(n[i + 1, :])) .> 1e-10)
        end
        return n[N + 1, :]
    else
        N_rev = Ns - N                  # num. of reverse recursions
        n = ones(Tn, N_rev, Ñs)
        for i = 1 : N_rev - 1
            n[i + 1, :] = (P[Ns - i + 1] / P[Ns - i]) * n[i, :] ./ expβϵμ
            # Truncate the values that are smaller than 10^-10
            n[i + 1, :] = n[i + 1, :] .* (abs.(real(n[i + 1, :])) .> 1e-10)
            n[i + 1, :] = 1 .- n[i + 1, :]
        end
        return n[N_rev, :]
    end
end

function second_order_corr(
    Ns::Int64, expβϵ::Array{T,1}, ni::Array{T,1}
) where {T<:FloatType}
    """
    Generate second-order correlation matrix, i.e., ⟨n_{i} n_{j}⟩ using the formula:
        ⟨n_{i} n_{j}⟩ = (n_{i}/expβϵ_{i} - n_{j}/expβϵ_{j}) / (1/expβϵ_{i} - 1/expβϵ_{j}).
    Practically, degenerate levels (expβϵ_{i} = expβϵ_{j}) should be super rare so we dont consider it here.
    """
    nij = zeros(T, Ns, Ns)
    expβϵ_inv = 1 ./ expβϵ
    for i = 2 : Ns
        niexpβϵ = ni[i] * expβϵ_inv[i]
        for j = 1 : i - 1
            njexpβϵ = ni[j] * expβϵ_inv[j]
            nij[i, j] = (niexpβϵ - njexpβϵ) / (expβϵ_inv[i] - expβϵ_inv[j])
            nij[j, i] = nij[i, j]
        end
    end
    return nij
end
