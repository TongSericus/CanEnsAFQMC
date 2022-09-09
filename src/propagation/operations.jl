"""
    All operations in the propagation
"""

function initial_propagation(
    auxfield::Matrix{Int64}, system::System, qmc::QMC; K = qmc.K
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

    if qmc.isLowrank
        F = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        F = [UDT(Ns), UDT(Ns)]
    else
        F = [UDR(Ns), UDR(Ns)]
    end

    B = [Matrix{Float64}(undef, Ns, Ns), Matrix{Float64}(undef, Ns, Ns)]
    MP = Cluster(Ns, K * 2)

    for i in 1 : K

        for j = 1 : qmc.stab_interval
            @views σ = auxfield[:, (i - 1) * qmc.stab_interval + j]
            singlestep_matrix!(B, σ, system)
            MP.B[i] = B[1] * MP.B[i]            # spin-up
            MP.B[K + i] = B[2] * MP.B[K + i]    # spin-down
        end

        qmc.isLowrank ? 
            F = [QR_lmul(MP.B[i], F[1], system.N[1], qmc.lrThld), QR_lmul(MP.B[K + i], F[2], system.N[2], qmc.lrThld)] :
            F = [QR_lmul(MP.B[i], F[1]), QR_lmul(MP.B[K + i], F[2])]

    end

    return F, MP

end

function full_propagation(MP::Cluster{T}, system::System, qmc::QMC) where T
    """
    Full propagation over the entire space-time auxiliary field 
    for test purposes
    """
    Ns = system.V

    if qmc.isLowrank
        F = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        F = [UDT(Ns), UDT(Ns)]
    else
        F = [UDR(Ns), UDR(Ns)]
    end

    for i in 1 : qmc.K
        qmc.isLowrank ?
            F = [QR_lmul(MP.B[i], F[1], system.N[1], qmc.lrThld), QR_lmul(MP.B[qmc.K + i], F[2], system.N[2], qmc.lrThld)] :
            F = [QR_lmul(MP.B[i], F[1]), QR_lmul(MP.B[qmc.K + i], F[2])]
    end
    return F
end

function partial_propagation(MP::Cluster{T}, system::System, qmc::QMC, a::Vector{Int64}) where T
    """
    Propagation over the space-and-partial-time field
    for calibration and test purposes
    """
    Ns = system.V

    if qmc.isLowrank
        F = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        F = [UDT(Ns), UDT(Ns)]
    else
        F = [UDR(Ns), UDR(Ns)]
    end

    for i in a
        qmc.isLowrank ?
            F = [QR_lmul(MP.B[i], F[1], system.N[1], qmc.lrThld), QR_lmul(MP.B[qmc.K + i], F[2], system.N[2], qmc.lrThld)] :
            F = [QR_lmul(MP.B[i], F[1]), QR_lmul(MP.B[qmc.K + i], F[2])]
    end
    return F
end

function update_matrices!(F0::UDT, Ft::UDT)
    F0.U .= Ft.U
    F0.D .= Ft.D
    F0.T .= Ft.T
end

function update_matrices!(F0::UDR, Ft::UDR)
    F0.U .= Ft.U
    F0.D .= Ft.D
    F0.R .= Ft.R
end

function update_matrices!(
    F0::Vector{UDR{T}}, Ft::Vector{UDR{T}}
) where {T<:FloatType}
    length(F0) == length(Ft) || @error "Mismatching Size"
    for i in 1 : length(F0)
        F0[i].U .= Ft[i].U
        F0[i].D .= Ft[i].D
        F0[i].R .= Ft[i].R
    end
end

function calibrate(system::System, qmc::QMC, cluster::Cluster{T}, cidx::Int64) where T
    """
    Factorization matrices need to be recalculated periodically

    # Arguments
    cidx -> cluster index
    """
    Ns = system.V

    if qmc.isLowrank
        FL = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        FL = [UDT(Ns), UDT(Ns)]
    else
        FL = [UDR(Ns), UDR(Ns)]
    end

    if cidx == qmc.K
        return FL
    else
        FL = partial_propagation(cluster, system, qmc, collect(cidx + 1 : qmc.K))
        return FL
    end
end

"""
    Repartition scheme
"""
function repartition(F::UDTlr{T}, γ::Float64) where {T<:FloatType}
    d = F.D
    l = length(F.t)
    # truncate from above
    Nu = div(2 * l, 3)
    while Nu > div(l, 3) && d[Nu + 1] / d[Nu] > γ
        Nu -= 1
    end
    Nu < div(l, 3) && return eigvals(F)
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
