"""
    Metropolis Sampling (MCMC)
"""

function initialize_walker_mcmc(system::System, qmc::QMC)
    """
    Initialize a walker with a random configuration
    """
    # initialize a random field configuration
    σfield = (rand(system.V, system.L) .< 0.5) .+ 1
    # initial propagation
    Q, D, T = qmc.lowrank ?
        full_propagation_lowrank(σfield, system, qmc) :
        full_propagation(σfield, system, qmc)
    # diagonalize the decomposition
    expβϵ = (
        eigvals(D[1] * T[1] * Q[1], sortby = abs),
        eigvals(D[2] * T[2] * Q[2], sortby = abs)
    )
    # fill the truncated slots with some super small numbers
    expβϵ = (
        vcat(1e-10 * ones(system.V - length(expβϵ[1])), expβϵ[1]),
        vcat(1e-10 * ones(system.V - length(expβϵ[2])), expβϵ[2])
    )
    # calculate the statistical weight
    Z = (
        recursion(system.V, system.N[1], expβϵ[1], false),
        recursion(system.V, system.N[2], expβϵ[2], false)
    )

    # construct the walker
    return walker = Walker(
        [Z[1], Z[2]],
        σfield,
        deepcopy(Q),
        deepcopy(D),
        deepcopy(T),
        0
    )

end

function calc_proposal_mcmc(
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
        recursion(system.V, system.N[1], expβϵ[1], false),
        recursion(system.V, system.N[2], expβϵ[2], false)
    )
    
    return TrialWalker(
        [Z[1], Z[2]], 
        [QDT[1][1], QDT[2][1]], 
        [QDT[1][2], QDT[2][2]], 
        [QDT[1][3], QDT[2][3]]
    )

end

function move!_mcmc(walker::Walker, system::System, time_index::Int64)
    """
    Move the walker to the next time slice,
    i.e. calculate B = B_{l-1}...B_1 * B_L...B_l * B_l^-1
    """
    Bl = singlestep_matrix(walker.auxfield[:, time_index], system)
    QRCP_update!(walker.Q[1], walker.D[1], walker.T[1], inv(Bl[1]), 'R')
    QRCP_update!(walker.Q[2], walker.D[2], walker.T[2], inv(Bl[2]), 'R')

end

function update_matrices!(
    Q0::Vector{T1}, D0::Vector{T2}, T0::Vector{T1},
    Q::Vector{T1}, D::Vector{T2}, T::Vector{T1}
    ) where {T1<:MatrixType, T2<:MatrixType}
    
    Q0[1] .= Q[1]
    Q0[2] .= Q[2]
    D0[1] .= D[1]
    D0[2] .= D[2]
    T0[1] .= T[1]
    T0[2] .= T[2]

end

@inline function flip!(auxfield::Array{Int64,2}, i::Int64, j::Int64)
    auxfield[i, j] = auxfield[i, j] % 2 + 1
end

function propagate!_mcmc(
    system::System, walker::Walker, 
    field_index::Tuple{Int64, Int64}, temp::Temp
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
    trialwalker = calc_proposal_mcmc(
        system, walker, walker.auxfield[:, field_index[1]]
    )
    # weight ratio between the new and the old configurations
    ratio = (trialwalker.weight[1] / walker.weight[1]) * (trialwalker.weight[2] / walker.weight[2])

    # heat bath sampling
    if rand() < abs(ratio) / (1 + abs(ratio))
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

function calibrate!_mcmc(system::System, qmc::QMC, walker::Walker, time_index::Int64)
    """
    Q, D, T matrices need to be recalculated periodically

    # Arguments
    time_index -> time slice index
    """
    shifted_field = circshift(walker.auxfield, (0, -time_index))
    Q, D, T = full_propagation(shifted_field, system, qmc)
    update_walker_matrices!(walker, Q, D, T)

    #expβϵ = (
    #        eigvals(D[1] * T[1] * Q[1], sortby = abs),
    #        eigvals(D[2] * T[2] * Q[2], sortby = abs)
    #)
    #Z = (
    #    recursion(system.V, system.N[1], expβϵ[1], false),
    #    recursion(system.V, system.N[2], expβϵ[2], false)
    #)
    #walker.weight .= [Z[1], Z[2]]

end

###### MCMC with Replica for Renyi-2 Entropy ######
function propagate!_replica(
    system::System,
    walker1::Walker, walker2::Walker,
    field_index::Tuple{Int64, Int64},
    temp1::Temp, temp2::Temp
    )

    # propose a flip at the jth site of the ith time slice
    flip!(walker1.auxfield, field_index[2], field_index[1])
    flip!(walker2.auxfield, field_index[2], field_index[1])
    # calculate the corresponding weight (partition function)
    trailwalker1 = calc_proposal_mcmc(
        system, walker1, walker1.auxfield[:, field_index[1]], field_index[2]
        )
    trailwalker2 = calc_proposal_mcmc(
        system, walker2, walker2.auxfield[:, field_index[1]], field_index[2]
        )
    # weight ratio between the new and the old configurations
    ratio1 = (trailwalker1.weight[1] / walker1.weight[1]) * (trailwalker1.weight[2] / walker1.weight[2])
    ratio2 = (trailwalker2.weight[1] / walker2.weight[1]) * (trailwalker2.weight[2] / walker2.weight[2])

    ### heat bath sampling ###
    prob = [1, ratio1, ratio2, ratio1 * ratio2]
    norm_prob = sum(prob)
    # cumulative distribution
    cum_prob = cumsum(prob / norm_prob)

    a = rand()
    config_index = 1
    while a > cum_prob[config_index]
        config_index += 1
    end
    if config_index == 1
        # reject both proposals, rollback the changes
        flip!(walker1.auxfield, field_index[2], field_index[1])
        flip!(walker2.auxfield, field_index[2], field_index[1])
    elseif config_index == 2
        # accept flipping walker 1, reject flipping walker 2
        walker1.weight .= [trailwalker1.weight[1], trailwalker1.weight[2]]
        update_matrices!(
            temp1.Q, temp1.D, temp1.T,
            trialwalker1.Q, trialwalker1.D, trialwalker1.T
        )
        flip!(walker2.auxfield, field_index[2], field_index[1])
    elseif config_index == 3
        # accept flipping walker 2, reject flipping walker 1
        flip!(walker1.auxfield, field_index[2], field_index[1])
        walker2.weight .= [trailwalker2.weight[1], trailwalker2.weight[2]]
        update_matrices!(
            temp2.Q, temp2.D, temp2.T,
            trialwalker2.Q, trialwalker2.D, trialwalker2.T
        )
    elseif config_index == 4
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
