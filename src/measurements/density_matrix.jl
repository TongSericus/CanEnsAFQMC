struct TempMats{Tn, Tp}
    D::AbstractMatrix{Tn}
    ninj::AbstractMatrix{Tn}
    P1::AbstractArray{Tp, 4}
    P2::AbstractArray{Tp, 4}
end

struct DensityMatrices{T, Tn, Tp}
    """
        Density matrices
        For two-body terms we just compute the elements along diagonal lines
    """
    # One-body: <c_i^{+} cj>
    Do::Matrix{T}

    # For two-body terms we only compute the elements along diagonal lines
    # Two-body: <c_i^{+} ci c_j^{+} cj>
    Dt::Matrix{T}

    # Temporal data to avoid extra allocations
    tmp::TempMats{Tn, Tp}
end

function DensityMatrices(system::System)
    """
    Initialize a DM
    """
    Ns = system.V

    system.isReal ? T = Float64 : T = ComplexF64

    tmp = TempMats(
        zeros(ComplexF64, Ns, Ns),
        zeros(ComplexF64, Ns, Ns),
        zeros(ComplexF64, Ns, Ns, Ns, Ns),
        zeros(ComplexF64, Ns, Ns, Ns, Ns)
    )

    DensityMatrices(
        zeros(T, Ns, Ns),
        zeros(T, Ns, Ns),
        tmp
    )
end

function fill_DM!(DM::DensityMatrices{T}, F::UDTlr, N::Int64; isReal::Bool=true, computeTwoBody::Bool=true) where {T<:Number}
    """
    Compute the elements of one-/two-body density matrices
    """
    # One-body elements
    λ, Pocc, invPocc = eigen(F)
    ni = occ_recursion(λ, N)
    mul!(DM.tmp.D, Pocc, Diagonal(ni) * invPocc)

    isReal ? copyto!(DM.Do, real(DM.tmp.D)) : copyto!(DM.Do, DM.tmp.D)

    computeTwoBody || return DM

    # Two-body elements
    second_order_corr(λ, ni, ninj = DM.tmp.ninj)
    compute_P(Pocc, invPocc, P1 = DM.tmp.P1, P2 = DM.tmp.P2)
    fill_Dt!(
        ni, DM.tmp.ninj, 
        Pocc, invPocc, 
        Dt = DM.tmp.D, 
        P1 = DM.tmp.P1, P2 = DM.tmp.P2
    )

    system.isReal ? copyto!(DM.Dt, real(DM.tmp.D)) : copyto!(DM.Dt, DM.tmp.D)
    
    return DM
end

function compute_P(
    P::AbstractMatrix{T}, invP::AbstractMatrix{T};
    L = size(P),
    P1 = zeros(T, L[2], L[2], L[1], L[1]),
    P2 = zeros(T, L[2], L[2], L[1], L[1])
) where {T<:Number}
    """
    Compute the transformation matrices
    """
    @inbounds for i = 1 : L[1]
        for j = 1 : L[1]
            for α = 1 : L[2]
                for β = 1 : L[2]
                    P1[β, α, j, i] = P[i, α] * invP[α, i] * P[j, β] * invP[β, j]
                    P2[β, α, j, i] = P[i, α] * invP[α, j] * P[j, β] * invP[β, i]
                end
            end
        end
    end

    return P1, P2
end

function fill_Dt!(
    ni::AbstractVector{T}, ninj::AbstractMatrix{T}, 
    P::AbstractMatrix{T}, invP::AbstractMatrix{T};
    L = size(P), Dt = zeros(T, L[1], L[1]),
    P1 = zeros(T, L[2], L[2], L[1], L[1]),
    P2 = zeros(T, L[2], L[2], L[1], L[1])
) where {T<:Number}
    """
    Compute the elements of the two-body density matrix
    """
    compute_P(P, invP, P1 = P1, P2 = P2)
    ninj_2 = ni .- ninj

    @inbounds for i = 1 : L[1]
        for j = i : L[1]
            s = 0
            for α = 1 : L[2]
                for β = 1 : L[2]
                    s += P1[β, α, j, i] * ninj[β, α]
                    s += P2[β, α, j, i] * ninj_2[β, α]
                end
            end
            Dt[j, i] = s
            Dt[i, j] = s
        end
    end

    return Dt
end
