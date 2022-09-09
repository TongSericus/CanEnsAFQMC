function measure_transition_probability(system_og::System, qmc_og::QMC, walker::Walker)
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1}
        
        system_og -> original system
        system_ext -> extended system
    """
    Lhalf= system_og.L

    F, dummy = initial_propagation(walker.auxfield[:, 1 : Lhalf], system_og, qmc_og, K = qmc_og.K)
    logZ = sum(calc_pf(system_og, F[1], F[2]))
    F, dummy = initial_propagation(walker.auxfield[:, Lhalf + 1 : end], system_og, qmc_og, K = qmc_og.K)
    logZ += sum(calc_pf(system_og, F[1], F[2]))

    p = exp(logZ - sum(walker.weight))
    return min(1, abs(p))
end

function measure_transition_probability(system_ext::System, qmc_ext::QMC, walker1::Walker, walker2::Walker)
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2}
    """
    auxfield = hcat(walker1.auxfield, walker2.auxfield)

    F, dummy = initial_propagation(auxfield, system_ext, qmc_ext, K = qmc_ext.K)
    logZ = sum(calc_pf(system_ext, F[1], F[2]))

    p = exp(logZ - sum(walker1.weight) - sum(walker2.weight))
    return min(1, abs(p))
end

function measure_transition_probability(system_og::System, qmc_og::QMC, walker::GCEWalker)
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1} in GCE
    """

    Lhalf= system_og.L
    expβμ = exp(system_og.β * system_og.μ)

    F, dummy = initial_propagation(walker.auxfield[:, 1 : Lhalf], system_og, qmc_og, K = qmc_og.K)
    F = [computeG(F[1], expβμ), computeG(F[1], expβμ)]
    G1 = [Matrix(F[1]), Matrix(F[2])]

    F, dummy = initial_propagation(walker.auxfield[:, Lhalf + 1 : end], system_og, qmc_og, K = qmc_og.K)
    F = [computeG(F[1], expβμ), computeG(F[1], expβμ)]
    G2 = [Matrix(F[1]), Matrix(F[2])]

    p = det(G2[1] * G1[1] + (G2[1] - I) * (G1[1] - I))
    p *= det(G2[2] * G1[2] + (G2[2] - I) * (G1[2] - I))
    
    return min(1, abs(1 / p))
end

function measure_transition_probability(system_ext::System, walker1::GCEWalker, walker2::GCEWalker)
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2} in GCE
    """
    G1 = unshiftG(walker1, system_ext);
    G2 = unshiftG(walker2, system_ext);

    p = det(G2[1] * G1[1] + (G2[1] - I) * (G1[1] - I))
    p *= det(G2[2] * G1[2] + (G2[2] - I) * (G1[2] - I))

    return min(1, abs(p))
end
