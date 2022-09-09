"""
MCMC with two copies of walker for Renyi-2 Entropy
"""
function heatbath_sampling(weights::Vector{Float64})
    """
    Simple heatbath sampling
    """
    norm = sum(weights)
    cum_prob = cumsum(weights / norm)
    u = rand()
    idx = 1
    while u > cum_prob[idx]
        idx += 1
    end
    return idx
end

function update_cluster!(
    walker1::Walker{W, T1, UDTlr{T1}, C},
    walker2::Walker{W, T2, UDTlr{T2}, C}, 
    system::System, qmc::QMC, 
    cidx::Int64, F1::Vector{UDTlr{T1}}, F2::Vector{UDTlr{T2}}
) where {W<:FloatType, T1<:FloatType, T2<:FloatType, C}
    """
    cidx -> cluster index
    """
    k = qmc.stab_interval
    Bl1 = Cluster(system.V, 2 * k)
    Bl2 = Cluster(system.V, 2 * k)

    U1 = [F1[1].U, F1[2].U]
    R1 = [F1[1].T * walker1.cluster.B[cidx], F1[2].T * walker1.cluster.B[qmc.K + cidx]]

    U2 = [F2[1].U, F2[2].U]
    R2 = [F2[1].T * walker2.cluster.B[cidx], F2[2].T * walker2.cluster.B[qmc.K + cidx]]

    for i in 1 : k
        σ1 = @view walker1.auxfield[:, (cidx - 1) * k + i]
        Bl1.B[i], Bl1.B[k + i] = singlestep_matrix(σ1, system)
        R1 .*= [inv(Bl1.B[i]), inv(Bl1.B[k + i])]

        σ2 = @view walker2.auxfield[:, (cidx - 1) * k + i]
        Bl2.B[i], Bl2.B[k + i] = singlestep_matrix(σ2, system)
        R2 .*= [inv(Bl2.B[i]), inv(Bl2.B[k + i])]

        for j in 1 : system.V
            σ1[j] *= -1
            Z1 = calc_trial(
                σ1, system, qmc,
                UDTlr(U1[1], F1[1].D, R1[1], F1[1].t), 
                UDTlr(U1[2], F1[2].D, R1[2], F1[2].t)
            )

            σ2[j] *= -1
            Z2 = calc_trial(
                σ2, system, qmc,
                UDTlr(U2[1], F2[1].D, R2[1], F2[1].t), 
                UDTlr(U2[2], F2[2].D, R2[2], F2[2].t)
            )

            r1 = sum(Z1) - sum(walker1.weight)
            r1 = abs(exp(r1))
            r2 = sum(Z2) - sum(walker2.weight)
            r2 = abs(exp(r2))
            
            idx = heatbath_sampling([1, r1, r2, r1 * r2])

            if idx == 1
                # reject both trials
                σ1[j] *= -1
                σ2[j] *= -1
            elseif idx == 2
                # accept trial 1 but reject trial 2
                walker1.weight .= Z1
                σ2[j] *= -1
            elseif idx == 3
                # reject trial 1 but accept trial 1
                σ1[j] *= -1
                walker2.weight .= Z2
            elseif idx == 4
                # accept both trials
                walker1.weight .= Z1
                walker2.weight .= Z2
            end
        end
        Bl1.B[i], Bl1.B[k + i] = singlestep_matrix(σ1, system)
        U1 = [Bl1.B[i], Bl1.B[k + i]] .* U1

        Bl2.B[i], Bl2.B[k + i] = singlestep_matrix(σ2, system)
        U2 = [Bl2.B[i], Bl2.B[k + i]] .* U2
    end

    walker1.cluster.B[cidx] = prod(Bl1, k : -1 : 1)
    walker1.cluster.B[qmc.K + cidx] = prod(Bl1, 2*k : -1 : k+1)

    walker2.cluster.B[cidx] = prod(Bl2, k : -1 : 1)
    walker2.cluster.B[qmc.K + cidx] = prod(Bl2, 2*k : -1 : k+1)

    return nothing
end

function sweep!(
    system::System, qmc::QMC, 
    walker1::Walker{W, T1, UDTlr{T1}, C}, 
    walker2::Walker{W, T2, UDTlr{T2}, C};
    tmpL1 = deepcopy(walker1.F),
    tmpR1 = [UDTlr(system.V), UDTlr(system.V)],
    tmpL2 = deepcopy(walker2.F),
    tmpR2 = [UDTlr(system.V), UDTlr(system.V)]
) where {W<:FloatType, T1<:FloatType, T2<:FloatType, C}
    """
    Sweep two copies of walker over the entire space-time lattice
    """
    N = system.N
    calib_counter = 0
    for cidx in 1 : qmc.K
        tmpL1[1] = QR_rmul(tmpL1[1], inv(walker1.cluster.B[cidx]), N[1], qmc.lrThld)
        tmpL1[2] = QR_rmul(tmpL1[2], inv(walker1.cluster.B[qmc.K + cidx]), N[2], qmc.lrThld)

        tmpL2[1] = QR_rmul(tmpL2[1], inv(walker2.cluster.B[cidx]), N[1], qmc.lrThld)
        tmpL2[2] = QR_rmul(tmpL2[2], inv(walker2.cluster.B[qmc.K + cidx]), N[2], qmc.lrThld)

        calib_counter += 1
        if calib_counter == qmc.update_interval
            tmpL1 = calibrate(system, qmc, walker1.cluster, cidx)
            tmpL2 = calibrate(system, qmc, walker2.cluster, cidx)
            calib_counter = 0
        end

        F1 = [
            QR_merge(tmpR1[1], tmpL1[1], N[1], qmc.lrThld), 
            QR_merge(tmpR1[2], tmpL1[2], N[2], qmc.lrThld)
        ]
        F2 = [
            QR_merge(tmpR2[1], tmpL2[1], N[1], qmc.lrThld), 
            QR_merge(tmpR2[2], tmpL2[2], N[2], qmc.lrThld)
        ]
        update_cluster!(walker1, walker2, system, qmc, cidx, F1, F2)
        
        tmpR1[1] = QR_lmul(walker1.cluster.B[cidx], tmpR1[1], N[1], qmc.lrThld)
        tmpR1[2] = QR_lmul(walker1.cluster.B[qmc.K + cidx], tmpR1[2], N[2], qmc.lrThld)

        tmpR2[1] = QR_lmul(walker2.cluster.B[cidx], tmpR2[1], N[1], qmc.lrThld)
        tmpR2[2] = QR_lmul(walker2.cluster.B[qmc.K + cidx], tmpR2[2], N[2], qmc.lrThld)
    end

    walker1 = Walker{W, T1, UDTlr{T1}, C}(walker1.weight, walker1.auxfield, tmpR1, walker1.cluster)
    walker2 = Walker{W, T2, UDTlr{T2}, C}(walker2.weight, walker2.auxfield, tmpR2, walker2.cluster)

    return walker1, walker2
end
