"""
    Monte Carlo Simulator for MCMC and Constraint Path
"""

function copy_matrices!(walker::Walker, temp::Temp, REV::Bool)

    if REV
        walker.Q[1] .= copy(temp.Q[1])
        walker.Q[2] .= copy(temp.Q[2])

        walker.D[1] .= copy(temp.D[1])
        walker.D[2] .= copy(temp.D[2])

        walker.T[1] .= copy(temp.T[1])
        walker.T[2] .= copy(temp.T[2])
    else
        temp.Q[1] .= copy(walker.Q[1])
        temp.Q[2] .= copy(walker.Q[2])

        temp.D[1] .= copy(walker.D[1])
        temp.D[2] .= copy(walker.D[2])

        temp.T[1] .= copy(walker.T[1])
        temp.T[2] .= copy(walker.T[2])
    end

end

function sweep!(system::System, qmc::QMC, walker::Walker, temp::Temp)
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
            calibrate!_mcmc(system, qmc, walker, l)
            walker.update_count = 0
        end
    end

end

function mc_metropolis(system::System, qmc::QMC, measure::GeneralMeasure)

    walker = initialize_walker_mcmc(system, qmc)
    temp = Temp(
            deepcopy(walker.Q),
            deepcopy(walker.D),
            deepcopy(walker.T)
    )

    ### Monte Carlo Sampling ###
    nk_array = zeros(Float64, length(measure.DFTmats), qmc.nsamples)
    ####### Warm-up Step #######
    for i = 1 : qmc.nwarmups
        # sweep the entire space-time lattice
        sweep!(system, qmc, walker, temp)
    end
    ####### Measure Step #######
    for i = 1 : qmc.nsamples
        for j = 1 : 3
            sweep!(system, qmc, walker, temp)
        end
        nk_array[:, i] = measurement_mcmc(system, measure, walker)
    end

    return nk_array

end

function sweep!_replica(
    system::System, qmc::QMC,
    walker1::Walker, walker2::Walker,
    temp1::Temp, temp2::Temp
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
            calibrate!_mcmc(system, qmc, walker1, l)
            calibrate!_mcmc(system, qmc, walker2, l)
            walker1.update_count = 0
        end
    end

end

function mc_replica(system::System, qmc::QMC, etg::EtgMeasure)

    # initialize two copies of walker
    walker1 = initialize_walker_mcmc(system, qmc)
    walker2 = initialize_walker_mcmc(system, qmc)
    temp1 = Temp(
            deepcopy(walker1.Q),
            deepcopy(walker1.D),
            deepcopy(walker1.T)
    )
    temp2 = Temp(
            deepcopy(walker2.Q),
            deepcopy(walker2.D),
            deepcopy(walker2.T)
    )

    ### Monte Carlo Sampling ###
    ####### Warm-up Step #######
    for i = 1 : qmc.nwarmups
        # sweep the entire space-time lattice
        sweep!_replica(system, qmc, walker1, walker2, temp1, temp2)
    end
    ####### Measurement Step #######
    expS2 = zeros(ComplexF64, qmc.nsamples)
    expS2n = zeros(ComplexF64, length(etg.k), length(etg.k), qmc.nsamples)
    for i = 1 : qmc.nsamples
        for j = 1 : 3
            sweep!_replica(system, qmc, walker1, walker2, temp1, temp2)
        end
        walker1_profile = construct_walker_profile(system, walker1)
        walker2_profile = construct_walker_profile(system, walker2)
        # spin-up sector
        expS2_up, expS2n_up = measure_renyi2_entropy(system, etg, 1, walker1_profile[1], walker2_profile[1])
        # spin-down sector
        expS2_dn, expS2n_dn = measure_renyi2_entropy(system, etg, 2, walker1_profile[2], walker2_profile[2])
        # merge
        expS2[i] = expS2_up * expS2_dn
        expS2n[:, :, i] = expS2n_up * expS2n_dn'
    end

    return expS2, expS2n
    
end

function mc_constrained(system::System, qmc::QMC)
    
    walker_list = initialize_walker_list(system, qmc)

    ### Monte Carlo Sampling ###
    update_count = 0
    for l = 1 : system.L

        # move to the next time slice
        for walker in walker_list
            move!_constrained(walker, system)
        end

        # propagate through sites
        for i = 1 : system.V
            for walker in walker_list
                propagate!_constrained(system, walker, (l, i))
            end
        end

        # rescale the weights of the walkers
        weights = [walker.weight[1] for walker in walker_list]
        total_weight = sum(weights)
        scale = total_weight / qmc.ntot_walkers
        weights = update_weights!(walker_list, scale)

        # periodically perform the population control and the calibration
        update_count += 1
        if update_count == qmc.update_interval
            for walker in walker_list
                calibrate!(system, qmc, walker, l)
            end
            # population control
            comb!(walker_list, weights)
            update_count = 0
        end

    end

    return walker_list

end
