"""
    Metropolis Sampling (MCMC)
"""
function calc_trial(
    σ::AbstractArray{Int64}, system::System, F1::UDR{T}, F2::UDR{T}
) where {T<:FloatType}
    """
    F1 and F2 represent spin-up and spin-down sectors respectively
    """
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U, Btmp[2] * F2.U]
    Ftmp = [UDR(Utmp[1], F1.D, F1.R), UDR(Utmp[2], F2.D, F2.R)]
    λ = [eigvals(Ftmp[1]), eigvals(Ftmp[2])]
    Z = [
        pf_recursion(system.V, system.N[1], λ[1]),
        pf_recursion(system.V, system.N[2], λ[2])
    ]

    system.isReal && return real(Z)
    return Z
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, F1::UDTlr{T}, F2::UDTlr{T}, γ::Float64
) where {T<:FloatType}
    N = system.N
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U[:, F1.t], Btmp[2] * F2.U[:, F2.t]]
    # compute Z with approximate form
    λocc, λf = repartition(UDTlr(Utmp[1], F1.D, F1.T, F1.t), N[1], γ)
    if length(λocc) == 0
        Z1 = pf_recursion(length(λf), N[1], λf)
    else
        Z1 = prod(λocc) * pf_recursion(length(λf), N[1] - length(λocc), λf) +
            (prod(λocc) - sum(λocc)) * pf_recursion(length(λf), N[1] - length(λocc) + 1, λf)
    end
    
    λocc, λf = repartition(UDTlr(Utmp[2], F2.D, F2.T, F2.t), N[2], γ)
    if length(λocc) == 0
        Z2 = pf_recursion(length(λf), N[2], λf)
    else
        Z2 = prod(λocc) * pf_recursion(length(λf), N[2] - length(λocc), λf) +
            (prod(λocc) - sum(λocc)) * pf_recursion(length(λf), N[2] - length(λocc) + 1, λf)
    end

    system.isReal && return real([Z1, Z2])
    return [Z1, Z2]
end

function update_cluster!(
    walker::Walker{T1, T2, UDR{T2}, S}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {T1<:FloatType, T2<:FloatType, T<:FloatType, S}
    """
    cidx -> cluster index
    """
    k = qmc.stab_interval
    U1, D1, R1 = F1
    U2, D2, R2 = F2
    R1 = R1 * walker.cluster.B[cidx]
    R2 = R2 * walker.cluster.B[qmc.K + cidx]
    Bl = Cluster(system.V, 2 * k)

    for i in 1 : k
        σ = @view walker.auxfield[:, (cidx - 1) * k + i]
        Bl.B[i], Bl.B[k + i] = singlestep_matrix(σ, system)
        R1 = R1 * inv(Bl.B[i])
        R2 = R2 * inv(Bl.B[k + i])

        for j in 1 : system.V
            σ[j] *= -1
            Z = calc_trial(σ, system, UDR(U1, D1, R1), UDR(U2, D2, R2))
            r = (Z[1] / walker.weight[1]) * (Z[2] / walker.weight[2])
            accept = abs(r) / (1 + abs(r))
            if rand() < accept
                walker.weight .= Z
            else
                σ[j] *= -1
            end
        end
        Bl.B[i], Bl.B[k + i] = singlestep_matrix(σ, system)
        U1 = Bl.B[i] * U1
        U2 = Bl.B[k + i] * U2
    end

    walker.cluster.B[cidx] = prod(Bl, k : -1 : 1)
    walker.cluster.B[qmc.K + cidx] = prod(Bl, 2*k : -1 : k+1)

    return nothing
end

function update_cluster!(
    walker::Walker{T1, T2, UDTlr{T2}, S}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {T1<:FloatType, T2<:FloatType, T<:FloatType, S}
    k = qmc.stab_interval
    U1, D1, R1 = F1
    U2, D2, R2 = F2
    R1 = R1 * walker.cluster.B[cidx]
    R2 = R2 * walker.cluster.B[qmc.K + cidx]
    Bl = Cluster(system.V, 2 * k)

    for i in 1 : k
        σ = @view walker.auxfield[:, (cidx - 1) * k + i]
        Bl.B[i], Bl.B[k + i] = singlestep_matrix(σ, system)
        R1 = R1 * inv(Bl.B[i])
        R2 = R2 * inv(Bl.B[k + i])

        for j in 1 : system.V
            σ[j] *= -1
            Z = calc_trial(
                σ, system, 
                UDTlr(U1, D1, R1, F1.t), UDTlr(U2, D2, R2, F2.t), 
                qmc.rpThld
            )
            r = (Z[1] / walker.weight[1]) * (Z[2] / walker.weight[2])
            accept = abs(r) / (1 + abs(r))
            if rand() < accept
                walker.weight .= Z
            else
                σ[j] *= -1
            end
        end
        Bl.B[i], Bl.B[k + i] = singlestep_matrix(σ, system)
        U1 = Bl.B[i] * U1
        U2 = Bl.B[k + i] * U2
    end

    walker.cluster.B[cidx] = prod(Bl, k : -1 : 1)
    walker.cluster.B[qmc.K + cidx] = prod(Bl, 2*k : -1 : k+1)

    return nothing
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker{T1, T2, UDR{T2}, C};
    tmpL = deepcopy(walker.F),
    tmpR = [UDR(system.V), UDR(system.V)]
) where {T1<:FloatType, T2<:FloatType, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    calib_counter = 0
    for cidx in 1 : qmc.K
        QR_rmul!(tmpL[1], inv(walker.cluster.B[cidx]))
        QR_rmul!(tmpL[2], inv(walker.cluster.B[qmc.K + cidx]))

        calib_counter += 1
        if calib_counter == qmc.update_interval
            tmpL, tmpR = calibrate(system, qmc, walker.cluster, cidx)
            calib_counter = 0
        end

        update_cluster!(walker, system, qmc, cidx, QR_merge(tmpR[1], tmpL[1]), QR_merge(tmpR[2], tmpL[2]))
        
        QR_lmul!(walker.cluster.B[cidx], tmpR[1])
        QR_lmul!(walker.cluster.B[qmc.K + cidx], tmpR[2])
    end

    update_matrices!(walker.F, tmpR)

    return walker
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker{T1, T2, UDTlr{T2}, C};
    tmpL = deepcopy(walker.F),
    tmpR = [UDTlr(system.V), UDTlr(system.V)]
) where {T1<:FloatType, T2<:FloatType, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    N = system.N
    calib_counter = 0
    for cidx in 1 : qmc.K
        tmpL[1] = QR_rmul(tmpL[1], inv(walker.cluster.B[cidx]), N[1], qmc.lrThld)
        tmpL[2] = QR_rmul(tmpL[2], inv(walker.cluster.B[qmc.K + cidx]), N[2], qmc.lrThld)

        calib_counter += 1
        if calib_counter == qmc.update_interval
            tmpL, tmpR = calibrate(system, qmc, walker.cluster, cidx)
            calib_counter = 0
        end

        update_cluster!(
            walker, system, qmc, cidx, 
            QR_merge(tmpR[1], tmpL[1], N[1], qmc.lrThld), 
            QR_merge(tmpR[2], tmpL[2], N[2], qmc.lrThld)
        )

        tmpR[1] = QR_lmul(walker.cluster.B[cidx], tmpR[1], N[1], qmc.lrThld)
        tmpR[2] = QR_lmul(walker.cluster.B[qmc.K + cidx], tmpR[2], N[2], qmc.lrThld)
    end

    walker = Walker{T1, T2, UDTlr{T2}, C}(walker.weight, walker.auxfield, tmpR, walker.cluster)
    return walker
end
