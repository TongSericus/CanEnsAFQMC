"""
    Some math functions
"""
function sgn(a::T) where {T<:FloatType}
    return a / abs(a)
end

function fermilevel(expβϵ::Vector{T}, N::Int64) where {T<:FloatType}
    """
    Compute an approximate Fermi level
    """
    Ns = length(expβϵ)
    return sqrt(abs(expβϵ[Ns - N + 1] * expβϵ[Ns - N]))
end

function regularized_complement(a::T; cutoff::Float64 = 1e-10) where {T<:FloatType}
    abs(1 - a) < cutoff && return cutoff
    return 1 - a
end

function quick_rotation(expiφ::Vector{ComplexF64}, N::Int64, isConj::Bool = false)
    """
    Rotation on a unit complex circle
    """
    Ns = length(expiφ)
    expiφ_rotated = copy(expiφ)
    for i = 1 : length(expiφ) - 1
        expiφ_rotated[i] = expiφ[mod(i * N, Ns)]
    end

    isConj || return expiφ_rotated
    return conj(expiφ_rotated)
end

function poissbino(
    Ns::Int64, ϵ::Array{T,1};
    isNormalized::Bool = false,
    P = zeros(eltype(ϵ), Ns + 1, Ns),
    cutoff = 1e-10
) where {T<:Number}
    """
    A regularized version of the recursive calculation for the Poisson binomial
    distribution
    For details on how this is employed in the canonical ensemble calculations,
    see doi.org/10.1103/PhysRevResearch.2.043206

    # Argument
    ϵ -> eigenvalues
    """
    ν1 = zeros(T, Ns)
    ν2 = zeros(T, Ns)
    if isNormalized
        ν1 = ϵ .+ (abs.(ϵ) .< cutoff) * cutoff
        ν2 = 1 .- ϵ .+ (abs.(1 .- ϵ) .< cutoff) * cutoff
    else
        ν1 = ϵ ./ (1 .+ ϵ)
        ν2 = 1 ./ (1 .+ ϵ)
    end

    # Initialization
    P[1, 1] = ν2[1]
    P[2, 1] = ν1[1]
    # iteration over trials
    @inbounds for i = 2 : Ns
        P[1, i] = ν2[i] * P[1, i - 1]
        # iteration over number of successes
        for j = 2 : i + 1
            P[j, i] = ν2[i] * P[j, i - 1] + ν1[i] * P[j - 1, i - 1]
        end
    end

    return P
end

function sum_antidiagonal(A::AbstractMatrix)
    """
    Sum the matrix elements over anti-diagonal directions, including
    all super/sub ones
    """
    row, col = size(A)
    row == col || @error "Non-sqaure matrix"
    v = Vector{eltype(A)}()

    for i = -col + 1 : col - 1
        if i < 0
            push!(v, sum([A[j, col + 1 + i - j] for j = 1 : col + i]))
        elseif i > 0
            push!(v, sum([A[j, col + 1 + i - j] for j = 1 + i : col]))
        else
            push!(v, sum([A[j, col + 1 + i - j] for j = 1 : col]))
        end
    end

    return v
end