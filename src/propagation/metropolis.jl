"""
    Metropolis Sampling (MCMC) (for different factorizations)
"""
function calc_pf(
    system::System, F1::Factorization{T}, F2::Factorization{T}; 
    isRepart::Bool = false, rpThld::Float64 = 0.02
) where {T<:FloatType}
    if isRepart
        λocc, λf = repartition(F1, rpThld)
        Z1 = pf_recursion(length(F1.t), system.N[1], vcat(λf, λocc))

        λocc, λf = repartition(F2, rpThld)
        Z2 = pf_recursion(length(F2.t), system.N[2], vcat(λf, λocc))

        return [Z1, Z2]
    else
        λ = [eigvals(F1), eigvals(F2)]
        Z = [
            pf_recursion(length(λ[1]), system.N[1], λ[1]),
            pf_recursion(length(λ[2]), system.N[2], λ[2])
        ]

        return Z
    end
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, F1::UDR{T}, F2::UDR{T}
) where {T<:FloatType}
    """
    F1 and F2 represent spin-up and spin-down sectors respectively
    """
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U, Btmp[2] * F2.U]
    Ftmp = [UDR(Utmp[1], F1.D, F1.R), UDR(Utmp[2], F2.D, F2.R)]
    return calc_pf(system, Ftmp[1], Ftmp[2])
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, F1::UDT{T}, F2::UDT{T}
) where {T<:FloatType}
    """
    F1 and F2 represent spin-up and spin-down sectors respectively
    """
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U, Btmp[2] * F2.U]
    Ftmp = [UDT(Utmp[1], F1.D, F1.T), UDT(Utmp[2], F2.D, F2.T)]
    return calc_pf(system, Ftmp[1], Ftmp[2])
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, qmc::QMC, F1::UDTlr{T}, F2::UDTlr{T}
) where {T<:FloatType}
    """
    Compute the weight of a trial configuration using a potential repartition scheme
    """
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U[:, F1.t], Btmp[2] * F2.U[:, F2.t]]
    Ftmp = [UDTlr(Utmp[1], F1.D, F1.T, F1.t), UDTlr(Utmp[2], F2.D, F2.T, F2.t)]
    return calc_pf(system, Ftmp[1], Ftmp[2]; isRepart = qmc.isRepart, rpThld = qmc.rpThld)
end

function update_cluster!(
    walker::Walker{W, T, UDR{T}, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDR{T}, F2::UDR{T}
) where {W<:FloatType, T<:FloatType, C}
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
            r = sum(Z) - sum(walker.weight)
            r = abs(exp(r))
            p = r / (1 + r)
            if rand() < p
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
    walker::Walker{W, T, UDT{T}, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDT{T}, F2::UDT{T}
) where {W<:FloatType, T<:FloatType, C}
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
            Z = calc_trial(σ, system, UDT(U1, D1, R1), UDT(U2, D2, R2))
            r = sum(Z) - sum(walker.weight)
            r = abs(exp(r))
            p = r / (1 + r)
            if rand() < p
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
    walker::Walker{W, T, UDTlr{T}, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {W<:FloatType, T<:FloatType, C}
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
                σ, system, qmc,
                UDTlr(U1, D1, R1, F1.t), UDTlr(U2, D2, R2, F2.t), 
            )
            r = sum(Z) - sum(walker.weight)
            r = abs(exp(r))
            p = r / (1 + r)
            #p = abs(r)
            if p > 1 || rand() < p
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
    walker::Walker{W, T, UDR{T}, C};
    tmpL = deepcopy(walker.F),
    tmpR = [UDR(system.V), UDR(system.V)]
) where {W<:FloatType, T<:FloatType, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    calib_counter = 0
    for cidx in 1 : qmc.K
        QR_rmul!(tmpL[1], inv(walker.cluster.B[cidx]))
        QR_rmul!(tmpL[2], inv(walker.cluster.B[qmc.K + cidx]))

        calib_counter += 1
        if calib_counter == qmc.update_interval
            tmpL = calibrate(system, qmc, walker.cluster, cidx)
            calib_counter = 0
        end

        update_cluster!(walker, system, qmc, cidx, QR_merge(tmpR[1], tmpL[1]), QR_merge(tmpR[2], tmpL[2]))
        
        QR_lmul!(walker.cluster.B[cidx], tmpR[1])
        QR_lmul!(walker.cluster.B[qmc.K + cidx], tmpR[2])
    end

    return Walker{W, T, UDR{T}, C}(walker.weight, walker.auxfield, tmpR, walker.cluster)
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker{W, T, UDT{T}, C};
    tmpL = deepcopy(walker.F),
    tmpR = [UDT(system.V), UDT(system.V)]
) where {W<:FloatType, T<:FloatType, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    calib_counter = 0
    for cidx in 1 : qmc.K
        tmpL[1] = QR_rmul(tmpL[1], inv(walker.cluster.B[cidx]))
        tmpL[2] = QR_rmul(tmpL[2], inv(walker.cluster.B[qmc.K + cidx]))

        calib_counter += 1
        if calib_counter == qmc.update_interval
            tmpL = calibrate(system, qmc, walker.cluster, cidx)
            calib_counter = 0
        end

        update_cluster!(walker, system, qmc, cidx, QR_merge(tmpR[1], tmpL[1]), QR_merge(tmpR[2], tmpL[2]))
        
        tmpR[1] = QR_lmul(walker.cluster.B[cidx], tmpR[1])
        tmpR[2] = QR_lmul(walker.cluster.B[qmc.K + cidx], tmpR[2])
    end

    return Walker{W, T, UDT{T}, C}(walker.weight, walker.auxfield, tmpR, walker.cluster)
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker{W, T, UDTlr{T}, C};
    tmpL = deepcopy(walker.F),
    tmpR = [UDTlr(system.V), UDTlr(system.V)]
) where {W<:FloatType, T<:FloatType, C}
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
            tmpL = calibrate(system, qmc, walker.cluster, cidx)
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

    return Walker{W, T, UDTlr{T}, C}(walker.weight, walker.auxfield, tmpR, walker.cluster)
end
