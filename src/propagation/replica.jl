export sweep!_replica
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
    ind = 1
    while u > cum_prob[ind]
        ind += 1
    end
    return ind
end

function propagate!_replica(
    system::System, walker1::Walker, walker2::Walker,
    field_index::Tuple{Int64, Int64},
    tmp1::Vector{UDT{T1}}, tmp2::Vector{UDT{T2}}
) where {T1,T2<:FloatType}
    # propose a flip at the jth site of the ith time slice
    flip!(walker1.auxfield, field_index[2], field_index[1])
    flip!(walker2.auxfield, field_index[2], field_index[1])
    # calculate the corresponding weight (partition function)
    Z1, F1 = calc_trail_mcmc(
        system, walker1, walker1.auxfield[:, field_index[1]]
    )
    Z2, F2 = calc_trail_mcmc(
        system, walker2, walker2.auxfield[:, field_index[1]]
    )
    # weight ratio between the new and the old configurations
    ratio1 = (Z1[1] / walker1.weight[1]) * (Z1[2] / walker1.weight[2])
    ratio2 = (Z2[1] / walker2.weight[1]) * (Z2[2] / walker2.weight[2])

    ### heat bath sampling ###
    index = heatbath_sampling([1, abs(ratio1), abs(ratio2), abs(ratio1 * ratio2)])
    
    if index == 1
        # reject both trials, rollback the changes
        flip!(walker1.auxfield, field_index[2], field_index[1])
        flip!(walker2.auxfield, field_index[2], field_index[1])
    elseif index == 2
        # accept the trial move of walker 1
        walker1.weight .= Z1
        update_matrices!(tmp1[1], F1[1])
        update_matrices!(tmp1[2], F1[2])
        # reject the trial move of walker 2
        flip!(walker2.auxfield, field_index[2], field_index[1])
    elseif index == 3
        # reject the trial move of walker 1
        flip!(walker1.auxfield, field_index[2], field_index[1])
        # accept the trial move of walker 2
        walker2.weight .= Z2
        update_matrices!(tmp2[1], F2[1])
        update_matrices!(tmp2[2], F2[2])
    elseif index == 4
        # accept both trials
        walker1.weight .= Z1
        update_matrices!(tmp1[1], F1[1])
        update_matrices!(tmp1[2], F1[2])

        walker2.weight .= Z2
        update_matrices!(tmp2[1], F2[1])
        update_matrices!(tmp2[2], F2[2])
    end
end

function sweep!_replica(
    system::System, qmc::QMC,
    walker1::Walker, walker2::Walker,
    tmp1::Vector{UDT{T1}}, tmp2::Vector{UDT{T2}}
) where {T1,T2<:FloatType}
    """
    Sweep two copies of walker over the entire space-time lattice
    """
    update_count = 0

    for l = 1 : system.L
        update_matrices!(tmp1[1], walker1.F[1])
        update_matrices!(tmp1[2], walker1.F[2])
        update_matrices!(tmp2[1], walker2.F[1])
        update_matrices!(tmp2[2], walker2.F[2])
        # move the walkers to the next time slice
        move!_mcmc(walker1, system, l)
        move!_mcmc(walker2, system, l)
        for i = 1 : system.V
            # propagate through sites
            propagate!_replica(
                system, walker1, walker2,
                (l, i),
                tmp1, tmp2
            )
            if i == system.V
                # update the matrices at the last spatial site
                update_matrices!(walker1.F[1], tmp1[1])
                update_matrices!(walker1.F[2], tmp1[2])
                update_matrices!(walker2.F[1], tmp2[1])
                update_matrices!(walker2.F[2], tmp2[2])
            end
        end
        # periodically calibrate the walker
        update_count += 1
        if update_count == qmc.update_interval
            calibrate!(system, qmc, walker1, l)
            calibrate!(system, qmc, walker2, l)
            update_count = 0
        end
    end
end
