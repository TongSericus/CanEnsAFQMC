"""
    Canonical Ensemble Recursions

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number (number of recursions)
    expβϵ -> exponentiated spectrum, i.e., exp(-βϵ)
"""
function pf_recursion(
    expβϵ::Vector{T}, N::Int64;
    isReal::Bool = false,
    useDouble::Bool = false,
    Ns = length(expβϵ),
    P = zeros(eltype(expβϵ), Ns + 1, Ns)
) where {T<:Number}
    """
        Recursive calculation of the partition function
    """
    isReal || (expβϵ = complex(expβϵ))

    N == 0 && return convert(T, 0.0)
    N == 1 && return log(sum(expβϵ))
    N == Ns && return sum(log.(expβϵ))
    
    # rescale spectrum
    expβμ = fermilevel(expβϵ, N)
    expβϵμ = expβϵ / expβμ
    # map to higher precision
    useDouble && (expβϵμ = ComplexDF64.(expβϵμ))
    
    poissbino(Ns, expβϵμ, P=P)

    if N > Ns / 2
        # non-logarithmic version: Z(N) = P(N) / P(Ns) * expβμ^(N - Ns) * Z(Ns)
        logZ = log(P[N+1, Ns]) - log(P[Ns+1, Ns]) - (Ns - N)*log(expβμ) + sum(log.(expβϵ))
    else
        # non-logarithmic version: Z(N) = P(N) / P(0) * expβμ^N
        logP0 = -sum(log.(1 .+ expβϵμ))
        logZ = log(P[N+1, Ns]) - logP0 + N*log(expβμ)
    end

    return logZ
end

function occ_recursion(
    expβϵ::Vector{T}, N::Int64;
    isReal::Bool = false,
    Ns = length(expβϵ),
    P = zeros(eltype(expβϵ), Ns + 1, Ns)
) where {T<:Number}
    """
        Recursive calculation of the occupation number
    """
    isReal || (expβϵ = complex(expβϵ))

    N == 0 && return zeros(T, Ns)
    N == Ns && return ones(T, Ns)
    N == 1 && return expβϵ / sum(expβϵ)

    expβμ = fermilevel(expβϵ, N)
    expβϵμ = expβϵ / expβμ
    poissbino(Ns, expβϵμ, P=P)

    # num of energy levels below the Fermi level
    # use this formula to ensure complex conjugate pairs are in the same section
    N_below = sum(abs.(expβϵμ) .> 1)

    # separate recursions for occupancies of energy levels above/below the Fermi level
    @views n_above = occ_recursion_rescaled(Ns, N, expβϵμ[1 : Ns - N_below], P[:, Ns])
    @views n_below = occ_recursion_rescaled(Ns, N, expβϵμ[Ns - N_below + 1 : Ns], P[:, Ns], isReverse=true)
    # then concatenate
    n = vcat(n_above, n_below)

    return n
end

function occ_recursion_rescaled(
    Ns::Int64, N::Int64,
    expβϵμ::AbstractArray{T}, P::AbstractArray{T};
    isReverse::Bool = false
) where T
    """
        Level occupancy recursion using the rescaled spectrum
    """
    Ñs = length(expβϵμ)
    if !isReverse
        n = zeros(ComplexF64, N + 1, Ñs)
        for i = 2 : N
            n[i + 1, :] = (P[i] / P[i + 1]) * expβϵμ .* (1 .- n[i, :])
            # Truncate the values that are smaller than 10^-10
            n[i + 1, :] = n[i + 1, :] .* (abs.(real(n[i + 1, :])) .> 1e-10)
        end
        return n[N + 1, :]
    else
        # num. of reverse recursions
        N_rev = Ns - N
        n = ones(ComplexF64, N_rev, Ñs)
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
