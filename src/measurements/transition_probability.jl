"""
    Measure the transition ratio between ensembles

    system_og -> original ensemble
    system_ext -> extended ensemble
"""

function measure_transition_probability(system_og::System, qmc_og::QMC, walker::Walker)
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1}
    """
    Lhalf= system_og.L

    F, dummy = run_full_propagation(walker.auxfield[:, 1 : Lhalf], system_og, qmc_og, K = qmc_og.K)
    logZ = sum(calc_pf(system_og, F[1], F[2]))
    F, dummy = run_full_propagation(walker.auxfield[:, Lhalf + 1 : end], system_og, qmc_og, K = qmc_og.K)
    logZ += sum(calc_pf(system_og, F[1], F[2]))

    p = exp(logZ - sum(walker.weight))
    return min(1, abs(p))
end

function measure_transition_probability(system_ext::System, qmc_ext::QMC, walker1::Walker, walker2::Walker)
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2}
    """
    auxfield = hcat(walker1.auxfield, walker2.auxfield)

    F, dummy = run_full_propagation(auxfield, system_ext, qmc_ext, K = qmc_ext.K)
    logZ = sum(calc_pf(system_ext, F[1], F[2]))

    p = exp(logZ - sum(walker1.weight) - sum(walker2.weight))
    return min(1, abs(p))
end


function measure_transition_probability(
    system_og::System, qmc_og::QMC, 
    system_ext::System, qmc_ext::QMC, 
    walker::GCEWalker
)
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1} in GCE
    """
    Lhalf= system_og.L
    expβμ = exp(system_og.β * system_og.μ)

    F, dummy = run_full_propagation(walker.auxfield[:, 1 : Lhalf], system_og, qmc_og)
    ZA = calc_pf(F[1], expβμ) + calc_pf(F[2], expβμ)

    F, dummy = run_full_propagation(walker.auxfield[:, Lhalf + 1 : end], system_og, qmc_og)
    ZB = calc_pf(F[1], expβμ) + calc_pf(F[2], expβμ)

    F, dummy = run_full_propagation(walker.auxfield, system_ext, qmc_ext)
    Z = calc_pf(F[1], walker.expβμ) + calc_pf(F[2], walker.expβμ)

    p = real(ZA + ZB - Z)
    
    return min(1., exp(p))
end

function measure_transition_probability(
    system_og::System, qmc_og::QMC, 
    system_ext::System, qmc_ext::QMC, 
    walker1::GCEWalker, walker2::GCEWalker
)
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2} in GCE
    """
    expβμ = exp(system_ext.β * system_ext.μ)
    auxfield = hcat(walker1.auxfield, walker2.auxfield)

    F, dummy = run_full_propagation(auxfield, system_ext, qmc_ext)
    Z = calc_pf(F[1], expβμ) + calc_pf(F[2], expβμ)

    F, dummy = run_full_propagation(walker1.auxfield, system_og, qmc_og)
    ZA = calc_pf(F[1], walker1.expβμ) + calc_pf(F[2], walker1.expβμ)

    F, dummy = run_full_propagation(walker2.auxfield, system_og, qmc_og)
    ZB = calc_pf(F[1], walker2.expβμ) + calc_pf(F[2], walker2.expβμ)

    p = real(Z - ZA - ZB)

    return min(1, exp(p))
end
