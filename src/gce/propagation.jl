function calc_trial_gce(
    σ::AbstractArray{Int64}, system::System, F1::UDTlr{T}, F2::UDTlr{T}
) where {T<:FloatType}
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U[:, F1.t], Btmp[2] * F2.U[:, F2.t]]
    Ftmp = [UDTlr(Utmp[1], F1.D, F1.T, F1.t), UDTlr(Utmp[2], F2.D, F2.T, F2.t)]
    λ = [eigvals(Ftmp[1]), eigvals(Ftmp[2])]

    sgn1, logZ1 = calc_pf(system.β, system.μ, λ[1])
    sgn2, logZ2 = calc_pf(system.β, system.μ, λ[2])
    
    system.isReal && return real([sgn1, sgn2]), [logZ1, logZ2]
    return [sgn1, sgn2], [logZ1, logZ2]
end

function update_cluster!(
    walker::GCEWalker{W, T, UDTlr{T}, C}, system::System, qmc::QMC, 
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
            sgn, logZ = calc_trial_gce(
                σ, system,
                UDTlr(U1, D1, R1, F1.t), UDTlr(U2, D2, R2, F2.t), 
            )
            r = (logZ[1] - walker.logweight[1]) + (logZ[2] - walker.logweight[2])
            accept = exp(r) / (1 + exp(r))
            if rand() < accept
                walker.sgn .= sgn
                walker.logweight .= logZ
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
    walker::GCEWalker{W, T, UDTlr{T}, C};
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

    return GCEWalker{W, T, UDTlr{T}, C}(walker.sgn, walker.logweight, walker.auxfield, tmpR, walker.cluster)
end

"""
    GCE sweep with customized μ
"""

function calc_trial_gce(
    σ::AbstractArray{Int64}, system::System, μt::Float64, F1::UDTlr{T}, F2::UDTlr{T}
) where {T<:FloatType}
    Btmp = singlestep_matrix(σ, system)
    Utmp = [Btmp[1] * F1.U[:, F1.t], Btmp[2] * F2.U[:, F2.t]]
    Ftmp = [UDTlr(Utmp[1], F1.D, F1.T, F1.t), UDTlr(Utmp[2], F2.D, F2.T, F2.t)]
    λ = [eigvals(Ftmp[1]), eigvals(Ftmp[2])]

    sgn1, logZ1 = calc_pf(system.β, μt, λ[1])
    sgn2, logZ2 = calc_pf(system.β, μt, λ[2])
    
    system.isReal && return real([sgn1, sgn2]), [logZ1, logZ2]
    return [sgn1, sgn2], [logZ1, logZ2]
end

function update_cluster!(
    walker::GCEWalker{W, T, UDTlr{T}, C}, system::System, qmc::QMC, 
    μt::Float64, cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
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
            sgn, logZ = calc_trial_gce(
                σ, system, μt,
                UDTlr(U1, D1, R1, F1.t), UDTlr(U2, D2, R2, F2.t), 
            )
            r = (logZ[1] - walker.logweight[1]) + (logZ[2] - walker.logweight[2])
            accept = exp(r) / (1 + exp(r))
            if rand() < accept
                walker.sgn .= sgn
                walker.logweight .= logZ
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
    walker::GCEWalker{W, T, UDTlr{T}, C}, μt::Float64;
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
            walker, system, qmc, μt, cidx, 
            QR_merge(tmpR[1], tmpL[1], N[1], qmc.lrThld), 
            QR_merge(tmpR[2], tmpL[2], N[2], qmc.lrThld)
        )

        tmpR[1] = QR_lmul(walker.cluster.B[cidx], tmpR[1], N[1], qmc.lrThld)
        tmpR[2] = QR_lmul(walker.cluster.B[qmc.K + cidx], tmpR[2], N[2], qmc.lrThld)
    end

    return GCEWalker{W, T, UDTlr{T}, C}(walker.sgn, walker.logweight, walker.auxfield, tmpR, walker.cluster)
end