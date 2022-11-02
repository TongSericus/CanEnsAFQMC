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

function fill_DM!(
    DM::DensityMatrices{T, Tn, Tp}, F::UDTlr, N::Int64; 
    isReal::Bool=true, computeTwoBody::Bool=true
) where {T<:Number, Tn, Tp}
    """
    Compute the elements of one-/two-body density matrices
    """
    # One-body elements
    mat = Diagonal(F.D) * F.T * F.U
    λ, P = eigen!(mat, sortby=abs)
    P = F.U * P
    invP = inv(P)

    ni = occ_recursion(λ, N)
    mul!(DM.tmp.D, P, Diagonal(ni) * invP)

    isReal ? copyto!(DM.Do, real(DM.tmp.D)) : copyto!(DM.Do, DM.tmp.D)

    computeTwoBody || return DM

    # Two-body elements
    second_order_corr(λ, ni, ninj = DM.tmp.ninj)
    compute_P(P, invP, P1 = DM.tmp.P1, P2 = DM.tmp.P2)
    fill_Dt!(
        ni, DM.tmp.ninj, 
        P, invP, 
        Dt = DM.tmp.D, 
        P1 = DM.tmp.P1, P2 = DM.tmp.P2
    )

    isReal ? copyto!(DM.Dt, real(DM.tmp.D)) : copyto!(DM.Dt, DM.tmp.D)
    
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
    ni::AbstractVector{Tn}, ninj::AbstractMatrix{Tn}, 
    P::AbstractMatrix{Tp}, invP::AbstractMatrix{Tp};
    L = size(P), Dt = zeros(Tn, L[1], L[1]),
    P1 = zeros(Tp, L[2], L[2], L[1], L[1]),
    P2 = zeros(Tp, L[2], L[2], L[1], L[1])
) where {Tn<:Number, Tp<:Number}
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

### GCE Density Matrices ###
function fill_DM!(DM::DensityMatrices{T, Tn, Tp}, G::AbstractMatrix{T}) where {T<:Number, Tn, Tp}
    Ns = size(G)[1]

    # One-body
    copyto!(DM.Do, I - adjoint(G))

    # Two-body, Wick's theorem
    # <a_i^+ a_j a_k^+ a_l> = <a_i^+ a_j><a_k^+ a_l> + <a_i^+ a_l>(δ_{kj} - <a_k^+ a_j>)
    Do = DM.Do
    Dt = DM.Dt
    @inbounds for j in 1 : Ns
        for i in j : Ns
            Dt[i, j] = Do[i, i] * Do[j, j] + Do[i, j] * ((i == j) - Do[j, i])
            Dt[j, i] = Dt[i, j]
        end
    end

    return DM
end
