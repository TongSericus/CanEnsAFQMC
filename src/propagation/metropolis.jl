"""
    Metropolis Sampling (MCMC)
"""

function calc_trail_mcmc(
    system::System, walker::Walker, σ::Vector{Int64}
    )
    """
    Calculate the statistical weight when the (i,j)th field component is proposed
    Shared with Metropolis

    # Arguments
    σ -> auxiliary field at the jth time slice

    # Returns
    Z -> statistical weights of the configuration
    QDT -> matrix decompositions as the last site is updated
    """
    # generate Bi
    B = singlestep_matrix(σ, system)
    # calculate the decomposition of B_i * B_{i-1} *...* B_1 * B_T *...* B_T
    QDT = [
        QRCP_update(walker.Q[1], walker.D[1], walker.T[1], B[1], 'L'),
        QRCP_update(walker.Q[2], walker.D[2], walker.T[2], B[2], 'L')
    ]
    # diagonalize the decomposition
    expβϵ = [
        eigvals(QDT[1][2] * QDT[1][3] * QDT[1][1], sortby = abs),
        eigvals(QDT[2][2] * QDT[2][3] * QDT[2][1], sortby = abs)
    ]
    # calculate the statistical weight (partition function)
    Z = [
        pf_projection(system.V, system.N[1], expβϵ[1], system.expiφ, false),
        pf_projection(system.V, system.N[2], expβϵ[2], system.expiφ, false)
    ]
    
    return TrialWalker(
        Z, 
        [QDT[1][1], QDT[2][1]], 
        [QDT[1][2], QDT[2][2]], 
        [QDT[1][3], QDT[2][3]]
    )

end

function propagate!_mcmc(
    system::System, walker::Walker, 
    field_index::Tuple{Int64, Int64}, temp::MatDecomp
    )
    """
    Propagation at the jth site of the ith time slice

    # Arguments
    field_index[1] -> time slice index (i)
    field_index[2] -> site index (j)
    """
    # propose a flip at the jth site of the ith time slice
    flip!(walker.auxfield, field_index[2], field_index[1])
    # calculate the corresponding weight (partition function)
    trialwalker = calc_trail_mcmc(
        system, walker, walker.auxfield[:, field_index[1]]
    )
    # weight ratio between the new and the old configurations
    ratio = (trialwalker.weight[1] / walker.weight[1]) * (trialwalker.weight[2] / walker.weight[2])
    # heat-bath
    accept = abs(ratio) / (1 + abs(ratio))

    if rand() < accept
        # accept the proposal, update weights
        walker.weight .= [trialwalker.weight[1], trialwalker.weight[2]]
        # update the temporal matrix decompositions
        update_matrices!(temp.Q, temp.D, temp.T,
            trialwalker.Q, trialwalker.D, trialwalker.T
        )
    else
        # reject the proposal, rollback the change
        flip!(walker.auxfield, field_index[2], field_index[1])
    end
end

function sweep!(system::System, qmc::QMC, walker::Walker, temp::MatDecomp)
    """
    Sweep the walker over the entire space-time lattice
    """
    for l = 1 : system.L

        copy_matrices!(walker, temp, false)

        # move the walker to the next time slice
        move!_mcmc(walker, system, l)
        
        for i = 1 : system.V
            # propagate through sites
            propagate!_mcmc(
                system, walker, (l, i), temp
                )
            if i == system.V
                # update the matrices at the last spatial site
                copy_matrices!(walker, temp, true)
            end
        end

        # periodically calibrate the walker
        walker.update_count += 1
        if walker.update_count == qmc.update_interval
            calibrate!(system, qmc, walker, l)
            walker.update_count = 0
        end
    end
end
