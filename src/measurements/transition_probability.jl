"""
    Measure the transition ratio between ensembles

    system_og -> original ensemble
    system_ext -> extended ensemble
"""

function measure_TransitProb(system::System, qmc::QMC, walker::Walker)
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1}
    """
    Lhalf = div(system.L, 2)
    Khalf = div(qmc.K, 2)

    @views F, dummy1, dummy2 = run_full_propagation(walker.auxfield[:, 1 : Lhalf], system, qmc, K = Khalf)
    replica_weight = calc_pf(F[1], system.N[1]) + calc_pf(F[2], system.N[2])
    @views F, dummy1, dummy2 = run_full_propagation(walker.auxfield[:, Lhalf + 1 : end], system, qmc, K = Khalf)
    replica_weight += calc_pf(F[1], system.N[1]) + calc_pf(F[2], system.N[2])

    p = replica_weight - sum(walker.weight)
    sign_p = sign(exp(imag(p)im))
    return min(0, real(p)), sign_p
end

function measure_TransitProb(system::System, qmc::QMC, walker1::Walker, walker2::Walker)
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2}
    """
    auxfield = hcat(walker1.auxfield, walker2.auxfield)

    F, dummy1, dummy2 = run_full_propagation(auxfield, system, qmc, K = qmc.K * 2)
    connected_weight = calc_pf(F[1], system.N[1]) + calc_pf(F[2], system.N[2])

    p = connected_weight - sum(walker1.weight) - sum(walker2.weight)
    sign_p = sign(exp(imag(p)im))
    return min(0, real(p)), sign_p
end


function measure_TransitProb(
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

function measure_TransitProb(
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
