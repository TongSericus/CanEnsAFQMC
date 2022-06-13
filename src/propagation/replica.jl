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

function update_cluster!_replica(
    walker1::Walker, walker2::Walker, system::System, qmc::QMC, 
    cidx::Int64, F1::Vector{UDR{T1}}, F2::Vector{UDR{T2}}
) where {T1<:FloatType, T2<:FloatType}
    """
    cidx -> cluster index
    """
    k = qmc.stab_interval

    U1 = [F1[1].U, F1[2].U]
    R1 = [F1[1].R * walker1.cluster.B[cidx], F1[2].R * walker1.cluster.B[qmc.K + cidx]]
    Bl1 = Cluster(system.V, 2 * k)

    U2 = [F2[1].U, F2[2].U]
    R2 = [F2[1].R * walker2.cluster.B[cidx], F2[2].R * walker2.cluster.B[qmc.K + cidx]]
    Bl2 = Cluster(system.V, 2 * k)

    for i in 1 : k
        σ1 = @view walker1.auxfield[:, (cidx - 1) * k + i]
        Bl1.B[i], Bl1.B[k + i] = singlestep_matrix(σ1, system)
        R1 = [R1[1] * inv(Bl1.B[i]), R1[2] * inv(Bl1.B[k + i])]

        σ2 = @view walker2.auxfield[:, (cidx - 1) * k + i]
        Bl2.B[i], Bl2.B[k + i] = singlestep_matrix(σ2, system)
        R2 = [R2[1] * inv(Bl2.B[i]), R2[2] * inv(Bl2.B[k + i])]

        for j in 1 : system.V
            σ1[j] *= -1
            Z1 = calc_trial(σ1, system, UDR(U1[1], F1[1].D, R1[1]), UDR(U1[2], F1[2].D, R1[2]))

            σ2[j] *= -1
            Z2 = calc_trial(σ2, system, UDR(U2[1], F2[1].D, R2[1]), UDR(U2[2], F2[2].D, R2[2]))

            r1 = (Z1[1] / walker1.weight[1]) * (Z1[2] / walker1.weight[2])
            r2 = (Z2[1] / walker2.weight[1]) * (Z2[2] / walker2.weight[2])
            idx = heatbath_sampling([1, abs(r1), abs(r2), abs(r1 * r2)])

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
        U1 = [Bl1.B[i] * U1[1], Bl1.B[k + i] * U1[2]]

        Bl2.B[i], Bl2.B[k + i] = singlestep_matrix(σ2, system)
        U2 = [Bl2.B[i] * U2[1], Bl2.B[k + i] * U2[2]]
    end

    walker1.cluster.B[cidx] = prod(Bl1, k : -1 : 1)
    walker1.cluster.B[qmc.K + cidx] = prod(Bl1, 2*k : -1 : k+1)

    walker2.cluster.B[cidx] = prod(Bl2, k : -1 : 1)
    walker2.cluster.B[qmc.K + cidx] = prod(Bl2, 2*k : -1 : k+1)

    return nothing
end

function sweep!_replica(
    system::System, qmc::QMC, 
    walker1::Walker, walker2::Walker;
    tmpL1 = deepcopy(walker1.F),
    tmpR1 = [UDR(system.V), UDR(system.V)],
    tmpL2 = deepcopy(walker2.F),
    tmpR2 = [UDR(system.V), UDR(system.V)]
)
    """
    Sweep two copies of walker over the entire space-time lattice
    """
    calib_counter = 0
    for cidx in 1 : qmc.K
        QR_rmul!(tmpL1[1], inv(walker1.cluster.B[cidx]))
        QR_rmul!(tmpL1[2], inv(walker1.cluster.B[qmc.K + cidx]))

        QR_rmul!(tmpL2[1], inv(walker2.cluster.B[cidx]))
        QR_rmul!(tmpL2[2], inv(walker2.cluster.B[qmc.K + cidx]))

        calib_counter += 1
        if calib_counter == qmc.update_interval
            calibrate!(system, qmc, walker1.cluster,
                tmpL1, tmpR1, cidx
            )
            calibrate!(system, qmc, walker2.cluster,
                tmpL2, tmpR2, cidx
            )
            calib_counter = 0
        end

        F1 = [QR_merge(tmpR1[1], tmpL1[1]), QR_merge(tmpR1[2], tmpL1[2])]
        F2 = [QR_merge(tmpR2[1], tmpL2[1]), QR_merge(tmpR2[2], tmpL2[2])]
        update_cluster!_replica(walker1, walker2, system, qmc, cidx, F1, F2)
        
        QR_lmul!(walker1.cluster.B[cidx], tmpR1[1])
        QR_lmul!(walker1.cluster.B[qmc.K + cidx], tmpR1[2])

        QR_lmul!(walker2.cluster.B[cidx], tmpR2[1])
        QR_lmul!(walker2.cluster.B[qmc.K + cidx], tmpR2[2])
    end

    update_matrices!(walker1.F, tmpR1)
    update_matrices!(walker2.F, tmpR2)

    return walker1, walker2
end
