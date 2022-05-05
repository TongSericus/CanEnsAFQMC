"""
    Some math functions
"""
function sgn(a::T) where {T<:FloatType}
    return convert(ComplexF64, a / abs(a))
end

function fermilevel(expβϵ::Vector{T}, N::Int64) where {T<:FloatType}
    """
    Compute an approximate Fermi level
    """
    Ns = length(expβϵ)

    return (abs(expβϵ[Ns - N + 1]) + abs(expβϵ[Ns - N])) / 2
end

function regularized_complement(a::T, cutoff::Float64 = 1e-10) where {T<:FloatType}
    return (1 - a) + cutoff
end

function poissbino(
    Ns::Int64, ϵ::Array{T,1}, isNormalized::Bool
) where {T<:FloatType}
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
        ν1 = ϵ .+ (abs.(ϵ) .< 1e-10) * 1e-10
        ν2 = 1 .- ϵ .+ (abs.(1 .- ϵ) .< 1e-10) * 1e-10
    else
        ν1 = ϵ ./ (1 .+ ϵ)
        ν2 = 1 ./ (1 .+ ϵ)
    end
    # culmulative probability distribution
    P = zeros(T, Ns + 1, Ns)

    # Initialization
    P[1, 1] = ν2[1]
    P[2, 1] = ν1[1]
    # iteration over trials
    for i = 2 : Ns
        P[1, i] = ν2[i] * P[1, i - 1]
        # iteration over number of successes
        for j = 2 : i + 1
            P[j, i] = ν2[i] * P[j, i - 1] + ν1[i] * P[j - 1, i - 1]
        end
    end

    return P[:, Ns]
end

function sum_antidiagonal(A::AbstractMatrix)
    """
    Sum the matrix elements over anti-diagonal directions, including
    all super/sub ones
    """
    row, col = size(A)
    row == col || @error "Non-sqaure matrix"
    v = Vector{typeof(A[1])}()

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