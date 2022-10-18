"""
    All operations in the propagation
"""

function run_full_propagation(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC; 
    K = qmc.K, stab_interval = qmc.stab_interval, K_interval = qmc.K_interval
)
    """
        Stable propagation over the entire space-time auxiliary field
    from scratch

        # Arguments
        auxfield -> the entire Ns*L auxiliary field

        # Returns
        F -> matrix decompositions with Matrix(F) = BL...B1
    """
    Ns = system.V
    N = system.N
    ϵ = qmc.lrThld

    if qmc.isLowrank
        F = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        F = [UDT(Ns), UDT(Ns)]
    else
        F = [UDR(Ns), UDR(Ns)]
    end

    FC = Cluster(F[1], 2 * K)

    B = [Matrix{Float64}(undef, Ns, Ns), Matrix{Float64}(undef, Ns, Ns)]
    MP = Cluster(Ns, 2 * K)

    for i in 1 : K

        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * stab_interval + j]
            singlestep_matrix!(B[1], B[2], σ, system)
            MP.B[i] = B[1] * MP.B[i]            # spin-up
            MP.B[K + i] = B[2] * MP.B[K + i]    # spin-down
        end

        copyto!(FC.B[i], F[1])
        copyto!(FC.B[K + i], F[2])

        qmc.isLowrank ? 
            F = [QR_lmul(MP.B[i], F[1], N[1], ϵ), QR_lmul(MP.B[K + i], F[2], N[2], ϵ)] :
            F = [QR_lmul(MP.B[i], F[1]), QR_lmul(MP.B[K + i], F[2])]
    end

    return F, MP, FC
end

function run_full_propagation(MP::Cluster{T}, system::System, qmc::QMC; K = qmc.K) where T
    """
        Full propagation over the entire space-time auxiliary field 
    given the matrix cluster
    """
    Ns = system.V
    N = system.N
    ϵ = qmc.lrThld

    if qmc.isLowrank
        F = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        F = [UDT(Ns), UDT(Ns)]
    else
        F = [UDR(Ns), UDR(Ns)]
    end

    for i in 1 : K
        qmc.isLowrank ?
            F = [QR_lmul(MP.B[i], F[1], N[1], ϵ), QR_lmul(MP.B[K + i], F[2], N[2], ϵ)] :
            F = [QR_lmul(MP.B[i], F[1]), QR_lmul(MP.B[K + i], F[2])]
    end

    return F
end

"""
    Repartition scheme
"""
function repartition(F::UDTlr{T}, γ::Float64) where {T<:FloatType}
    d = F.D
    l = length(F.t)
    # truncate from above
    Nu = div(l, 3)
    γmin = d[Nu + 1] / d[Nu]

    for Nt = div(l, 3) : div(2 * l, 3)
        d[Nt + 1] / d[Nt] < γmin && (γmin = d[Nt + 1] / d[Nt]; Nu = Nt)
    end
    
    γmin > γ && return eigvals(F), []
    tocc = 1 : Nu
    tf = Nu + 1 : F.t.stop

    B = @views F.T[F.t, :] * F.U[:, F.t]
    P = @views B[tf, tocc] * inv(B[tocc, tocc])
    Mocc = @views B[tocc, tocc] * Diagonal(F.D[tocc]) + B[tocc, tf] * Diagonal(F.D[tf]) * P
    Mf = @views (B[tf, tf] - P * B[tocc, tf]) * Diagonal(F.D[tf])

    return eigvals(Mocc, sortby=abs), eigvals(Mf, sortby=abs)
end

function repartition(
    Ns::Int64, N::Int64, F::UDR{T}, γ::Float64, ϵ::Float64
) where {T<:FloatType}
    #normR = [norm(@view F.R[i, :])^2 for i in 1 : Ns]
    #d = F.D .* normR
    dsort = sortperm(F.D, rev = true)
    d = F.D[dsort]

    # truncation from above
    Nu = N - 1
    while Nu > 0 && d[Nu + 1] / d[Nu] > γ
        Nu -= 1
    end
    tocc = 1 :Nu
    t1 = @view dsort[tocc]

    # truncation from below
    dl = d[N] * ϵ
    Nl = N
    while Nl < Ns + 1 && d[Nl] > dl
        Nl += 1
    end
    tf = Nu + 1 : Nl - 1
    t2 = @view dsort[tf]

    t =  [t1; t2] # union of t1 and t2
    B = @views F.R[t, :] * F.U[:, t]
    P = @views B[tf, tocc] * inv(B[tocc, tocc])

    Mocc = @views B[tocc, tocc] * Diagonal(F.D[t1]) + B[tocc, tf] * Diagonal(F.D[t2]) * P
    Mf = @views (B[tf, tf] - P * B[tocc, tf]) * Diagonal(F.D[t2])

    return eigvals(Mocc, sortby=abs), eigvals(Mf, sortby=abs)
end

function repartition(
    Ns::Int64, N::Int64, F::UDT{T}, γ::Float64, ϵ::Float64
) where {T<:FloatType}
    d = F.D

    # truncation from above
    Nu = N - 1
    while Nu > 0 && d[Nu + 1] / d[Nu] > γ
        Nu -= 1
    end
    tocc = 1 : Nu

    # truncation from below
    dl = d[N] * ϵ
    Nl = N
    while Nl < Ns + 1 && d[Nl] > dl
        Nl += 1
    end
    tf = Nu + 1 : Nl - 1

    t = 1 : Nl - 1 # union of t1 and t2
    B = @views F.T[t, :] * F.U[:, t]
    P = @views B[tf, tocc] * inv(B[tocc, tocc])

    Mocc = @views B[tocc, tocc] * Diagonal(F.D[tocc]) + B[tocc, tf] * Diagonal(F.D[tf]) * P
    Mf = @views (B[tf, tf] - P * B[tocc, tf]) * Diagonal(F.D[tf])

    return eigvals(Mocc, sortby=abs), eigvals(Mf, sortby=abs)
end
