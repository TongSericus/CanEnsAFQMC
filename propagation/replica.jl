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
    system::System,
    walker1::Walker, walker2::Walker,
    field_index::Tuple{Int64, Int64},
    temp1::MatDecomp, temp2::MatDecomp
)

    # propose a flip at the jth site of the ith time slice
    flip!(walker1.auxfield, field_index[2], field_index[1])
    flip!(walker2.auxfield, field_index[2], field_index[1])
    # calculate the corresponding weight (partition function)
    trailwalker1 = calc_trail_mcmc(
        system, walker1, walker1.auxfield[:, field_index[1]], field_index[2]
    )
    trailwalker2 = calc_trail_mcmc(
        system, walker2, walker2.auxfield[:, field_index[1]], field_index[2]
    )
    # weight ratio between the new and the old configurations
    ratio1 = (trailwalker1.weight[1] / walker1.weight[1]) * (trailwalker1.weight[2] / walker1.weight[2])
    ratio2 = (trailwalker2.weight[1] / walker2.weight[1]) * (trailwalker2.weight[2] / walker2.weight[2])

    ### heat bath sampling ###
    index = heatbath_sampling([1, ratio1, ratio2, ratio1 * ratio2])
    
    if index == 1
        # reject both proposals, rollback the changes
        flip!(walker1.auxfield, field_index[2], field_index[1])
        flip!(walker2.auxfield, field_index[2], field_index[1])
    elseif index == 2
        # accept flipping walker 1, reject flipping walker 2
        walker1.weight .= [trailwalker1.weight[1], trailwalker1.weight[2]]
        update_matrices!(
            temp1.Q, temp1.D, temp1.T,
            trialwalker1.Q, trialwalker1.D, trialwalker1.T
        )
        flip!(walker2.auxfield, field_index[2], field_index[1])
    elseif index == 3
        # accept flipping walker 2, reject flipping walker 1
        flip!(walker1.auxfield, field_index[2], field_index[1])
        walker2.weight .= [trailwalker2.weight[1], trailwalker2.weight[2]]
        update_matrices!(
            temp2.Q, temp2.D, temp2.T,
            trialwalker2.Q, trialwalker2.D, trialwalker2.T
        )
    elseif index == 4
        # accept both proposals
        walker1.weight .= [trailwalker1.weight[1], trailwalker1.weight[2]]
        update_matrices!(
            temp1.Q, temp1.D, temp1.T,
            trialwalker1.Q, trialwalker1.D, trialwalker1.T
        )
        walker2.weight .= [trailwalker2.weight[1], trailwalker2.weight[2]]
        update_matrices!(
            temp2.Q, temp2.D, temp2.T,
            trialwalker2.Q, trialwalker2.D, trialwalker2.T
        )
    end

end

function sweep!_replica(
    system::System, qmc::QMC,
    walker1::Walker, walker2::Walker,
    temp1::MatDecomp, temp2::MatDecomp
)
    """
    Sweep two copies of walker over the entire space-time lattice
    """
    for l = 1 : system.L

        copy_matrices!(walker1, temp1, false)
        copy_matrices!(walker2, temp2, false)

        # move the walkers to the next time slice
        move!_mcmc(walker1, system, l)
        move!_mcmc(walker2, system, l)

        for i = 1 : system.V
            # propagate through sites
            propagate!_replica(
                system, walker1, walker2,
                (l, i),
                temp1, temp2
            )
            if i == system.V
                # update the matrices at the last spatial site
                copy_matrices!(walker1, temp1, true)
                copy_matrices!(walker2, temp2, true)
            end
        end

        # periodically calibrate the walker
        walker1.update_count += 1
        if walker1.update_count == qmc.update_interval
            calibrate!(system, qmc, walker1, l)
            calibrate!(system, qmc, walker2, l)
            walker1.update_count = 0
        end
    end

end
