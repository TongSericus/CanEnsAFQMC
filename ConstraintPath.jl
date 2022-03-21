"""
    Constraint Path Sampling
"""

function initialize_walker_list(system::System, qmc::QMC)
    """
    Initialize an ensemble of identical walkers
    """
    walker_list = Vector{Walker}()

    σfield = zeros(Int64, system.V, system.L)
    Q, D, T = full_propagation(σfield, system, qmc)
    # assume the trial propagator does not distinguish spins
    expβϵ = (
        eigvals(D[1] * T[1] * Q[1], sortby = abs),
        eigvals(D[2] * T[2] * Q[2], sortby = abs)
        )
    Z = (
        recursion(system.V, system.N[1], expβϵ[1], false),
        recursion(system.V, system.N[2], expβϵ[2], false)
        )

    walker = Walker(
        [1.0, 1.0],
        [Z[1], Z[2]],
        σfield,
        deepcopy(Q),
        deepcopy(D),
        deepcopy(T)
    )

    for i = 1 : qmc.ntot_walkers
        push!(walker_list, deepcopy(walker))
    end

    return walker_list

end

function calc_proposal_constrained(
    system::System, walker::Walker, 
    σ::Vector{Int64}, site_index::Int64)
    """
    Calculate the statistical weight when the (i,j)th field component is proposed
    Shared with Metropolis

    # Arguments
    σ -> auxiliary field at the jth time slice
    RETURN_ALL -> if true, return all information regardless of the walker's position1

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
    
    # matrix decompositions would also be returned if the walker reaches the next time slice or forced
    if site_index == system.V
        return Z[1], Z[2], (QDT[1][1], QDT[2][1]), (QDT[1][2], QDT[2][2]), (QDT[1][3], QDT[2][3])
    else
        return Z[1], Z[2]
    end

end

function update_walker_matrices!(
    walker::Walker, 
    Q::Vector{T1}, D::Vector{T2}, T::Vector{T1}
    ) where {T1<:MatrixType, T2<:MatrixType}
    
    walker.Q[1] .= Q[1]
    walker.Q[2] .= Q[2]
    walker.D[1] .= D[1]
    walker.D[2] .= D[2]
    walker.T[1] .= T[1]
    walker.T[2] .= T[2]

end

function propagate!_constrained(
    system::System, walker::Walker, field_index::Tuple{Int64, Int64}
    )
    """
    Propagation at the jth site of the ith time slice

    # Arguments
    field_index[1] -> time slice index (i)
    field_index[2] -> site index (j)
    """
    σ = walker.auxfield[:, field_index[1]]
    
    # propose σi[j] = +1
    σ[field_index[2]] = 1
    proposal_1 = calc_proposal_constrained(system, walker, σ, field_index[2])
    # propose σi[j] = -1
    σ[field_index[2]] = 2
    proposal_2 = calc_proposal_constrained(system, walker, σ, field_index[2])

    # weight ratio of the two proposed configurations
    prob = (
        proposal_1[1] / walker.Pl[1] * (proposal_1[2] / walker.Pl[2]),
        proposal_2[1] / walker.Pl[1] * (proposal_2[2] / walker.Pl[2])
        )
    # discard the negative weight
    prob = (prob .> 0) .* prob
    norm_prob = sum(prob)

    if norm_prob > 0
        if rand() < prob[1] / norm_prob
            # accept σi[j] = +1
            walker.auxfield[field_index[2], field_index[1]] = 1
            walker.Pl[1] = proposal_1[1]
            walker.Pl[2] = proposal_1[2]
            walker.weight .*= norm_prob
            #walker.weight .*= prob[1]
            if field_index[2] == system.V
                # at the last spatial site, also update the matrices
                update_walker_matrices!(walker, proposal_1[3], proposal_1[4], proposal_1[5])
            end
        else
            # accept σi[j] = -1
            walker.auxfield[field_index[2], field_index[1]] = 2
            walker.Pl[1] = proposal_2[1]
            walker.Pl[2] = proposal_2[2]
            walker.weight .*= norm_prob
            #walker.weight .*= prob[2]
            if field_index[2] == system.V
                # at the last spatial site, also update the matrices
                update_walker_matrices!(walker, proposal_2[3], proposal_2[4], proposal_2[5])
            end
        end
    else
        walker.weight .= [0.0, 0.0]
    end

end

function move!_constrained(walker::Walker, system::System)
    """
    Move the walker to the next time slice,
    i.e. calculate B = Bl...B1 * BT...BT * (BT)^-1
    """
    QRCP_update!(walker.Q[1], walker.D[1], walker.T[1], system.BT_inv, 'R')
    QRCP_update!(walker.Q[2], walker.D[2], walker.T[2], system.BT_inv, 'R')

end

function calibrate!(system::System, qmc::QMC, walker::Walker, time_index::Int64)
    """
    Q, D, T matrices need to be recalculated periodically

    # Arguments
    time_index -> time slice index
    """
    shifted_field = circshift(walker.auxfield, (0, -time_index))
    Q, D, T = full_propagation(shifted_field, system, qmc)
    update_walker_matrices!(walker, Q, D, T)

end

function update_weights!(walker_list::Vector{Walker}, scale::Float64)
    """
    Rescale the weights of all walkers by the same number
    """
    weights = Vector{Float64}()

    for walker in walker_list
        walker.weight ./= scale
        push!(weights, walker.weight[1])
    end
    
    return weights
end

function comb!(walker_list::Vector{Walker}, weights::Vector{Float64})
    """
    Adjust the distribution of walkers based on their weights (comb method)
    See doi.org/10.1103/PhysRevE.80.046704
    """
    num_walkers = length(weights)
    total_weight = sum(weights)
    # cumulative weights
    cum_weights = cumsum(weights)
    ξ = rand()
    # construct the comb
    comb = [(k + ξ - 1) * total_weight / num_walkers for k = 1 : num_walkers]
    comb_count = zeros(Int64, num_walkers)

    # clone the ith walker if cum_weights[i - 1] < comb < cum_weights[i]
    comb_index = 1
    weight_index = 1
    while comb_index <= length(comb)
        if comb[comb_index] < cum_weights[weight_index]
            comb_count[weight_index] += 1
            comb_index += 1
        else
            weight_index += 1
        end
    end

    kill_list = findall(x -> x == 0, comb_count)
    clone_list = findall(x -> x > 1, comb_count)
    # replace the walkers in the kill list with the walkers in the clone list
    clone_index = 1
    for i in kill_list
        if comb_count[clone_list[clone_index]] == 1
            clone_index += 1
        end
        walker_list[i] = deepcopy(walker_list[clone_list[clone_index]])
        comb_count[clone_list[clone_index]] -= 1
    end

end
