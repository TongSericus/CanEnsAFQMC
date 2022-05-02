"""
    Metropolis Sampling in the Grand Canonical Ensemble
"""

function initialize_walker_gce(system::System, qmc::QMC)
    """
    Initialize a walker with a random configuration
    """
    # initialize a random field configuration
    σfield = (rand(system.V, system.L) .< 0.5) .+ 1
    # initial propagation
    Q, D, T = full_propagation(σfield, system, qmc)
    # diagonalize the decomposition
    expβϵ = (
        eigvals(D[1] * T[1] * Q[1], sortby = abs),
        eigvals(D[2] * T[2] * Q[2], sortby = abs)
    )
    # calculate the GCE statistical weight
    Z = (
        real(prod(1 .+ system.expβμ * expβϵ[1])),
        real(prod(1 .+ system.expβμ * expβϵ[2]))
    )

    # construct the walker
    return Walker(
        [Z[1], Z[2]],
        σfield,
        deepcopy(Q),
        deepcopy(D),
        deepcopy(T),
        0
    )

end

function calc_proposal_gce(
    system::System, walker::Walker, σ::Vector{Int64}
    )
    """
    Calculate the statistical weight when the (i,j)th field component is proposed

    # Arguments
    σ -> auxiliary field at the jth time slice

    # Returns
    Z -> statistical weights of the configuration
    QDT -> matrix decompositions as the last site is updated
    """
    # generate Bi
    B = singlestep_matrix(σ, system)
    # calculate the decomposition of B_i * B_{i-1} *...* B_1 * B_T *...* B_T
    QDT = (
        QRCP_update(walker.Q[1], walker.D[1], walker.T[1], B[1], 'L'),
        QRCP_update(walker.Q[2], walker.D[2], walker.T[2], B[2], 'L')
    )
    # diagonalize the decomposition
    expβϵ = (
        eigvals(QDT[1][2] * QDT[1][3] * QDT[1][1], sortby = abs),
        eigvals(QDT[2][2] * QDT[2][3] * QDT[2][1], sortby = abs)
    )
    # calculate the statistical weight (partition function)
    Z = (
        real(prod(1 .+ system.expβμ * expβϵ[1])),
        real(prod(1 .+ system.expβμ * expβϵ[2]))
    )
    
    return TrialWalker(
        [Z[1], Z[2]],
        [QDT[1][1], QDT[2][1]],
        [QDT[1][2], QDT[2][2]],
        [QDT[1][3], QDT[2][3]]
    )

end

function propagate!_gce(
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
    trialwalker = calc_proposal_gce(
        system, walker, walker.auxfield[:, field_index[1]]
    )
    # weight ratio between the new and the old configurations
    ratio = (trialwalker.weight[1] / walker.weight[1]) * (trialwalker.weight[2] / walker.weight[2])

    # heat bath sampling
    if rand() < abs(ratio) / (1 + abs(ratio))
        # accept the proposal, update weights
        walker.weight .= [trialwalker.weight[1], trialwalker.weight[2]]
        # update the oral matrix decompositions
        update_matrices!(temp.Q, temp.D, temp.T,
                        trialwalker.Q, trialwalker.D, trialwalker.T
        )
    else
        # reject the proposal, rollback the change
        flip!(walker.auxfield, field_index[2], field_index[1])
    end

end

"""
    Measure Observables
"""

function measurement_mcmc_gce(system::System, measure::GeneralMeasure, walker::Walker)
    """
    Measurements in MCMC
    """
    # construct the density matrix
    walker_eigen = (
            eigen(walker.D[1] * walker.T[1] * walker.Q[1], sortby = abs),
            eigen(walker.D[2] * walker.T[2] * walker.Q[2], sortby = abs)
    )
    n = (
        (system.expβμ * walker_eigen[1].values) ./ (1 .+ system.expβμ * walker_eigen[1].values),
        (system.expβμ * walker_eigen[2].values) ./ (1 .+ system.expβμ * walker_eigen[2].values)
    )
    P = (
        walker.Q[1] * walker_eigen[1].vectors,
        walker.Q[2] * walker_eigen[2].vectors
    )
    G = (
        P[1] * Diagonal(n[1]) * inv(P[1]),
        P[2] * Diagonal(n[2]) * inv(P[2])
    )

    # measure momentum distribution
    nk = measure_momentum_dist(system, measure, G)

    return real(nk[1] .+ nk[2]) / 2

end

"""
    GCE Monte Carlo Simulator
"""

function sweep!_gce(system::System, qmc::QMC, walker::Walker, temp::MatDecomp)
    """
    Sweep the walker over the entire space-time lattice
    """
    for l = 1 : system.L

        copy_matrices!(walker, temp, false)

        # move the walker to the next time slice
        move!_mcmc(walker, system, l)
        
        for i = 1 : system.V
            # propagate through sites
            propagate!_gce(
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
            calibrate!_mcmc(system, qmc, walker, l)
            walker.update_count = 0
        end
    end

end

function mc_metropolis_gce(system::System, qmc::QMC, measure::GeneralMeasure)

    walker = initialize_walker_gce(system, qmc)
    temp = QDT(
        deepcopy(walker.Q),
        deepcopy(walker.D),
        deepcopy(walker.T)
    )

    ### Monte Carlo Sampling ###
    nk_array = zeros(Float64, length(measure.DFTmats), qmc.nsamples)
    ####### Warm-up Step #######
    for i = 1 : qmc.nwarmups
        # sweep the entire space-time lattice
        sweep!_gce(system, qmc, walker, temp)
    end
    ####### Measure Step #######
    for i = 1: qmc.nsamples
        for j = 1 : 2
            sweep!_gce(system, qmc, walker, temp)
        end
        nk_array[:, i] = measurement_mcmc_gce(system, measure, walker)
    end
    
    return nk_array

end
