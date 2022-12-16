"""
    Measure the transition ratio between ensembles
"""

function measure_TransitProb(system_og::System, qmc_og::QMC, walker::Walker)
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1}
    """
    N = system_og.N
    Lhalf = system_og.L
    P = walker.tempdata.P

    ws = walker.ws

    replica_weight = 0.0
    p_sign = 1.0

    @views F, dummy1, dummy2 = run_full_propagation(walker.auxfield[:, 1 : Lhalf], system_og, qmc_og, ws)
    tmp = compute_PF(F[1], N[1], PMat = P)
    replica_weight += tmp[1]
    p_sign *= tmp[2]
    tmp = compute_PF(F[2], N[2], PMat = P)
    replica_weight += tmp[1]
    p_sign *= tmp[2]

    @views F, dummy1, dummy2 = run_full_propagation(walker.auxfield[:, Lhalf + 1 : end], system_og, qmc_og, ws)
    tmp = compute_PF(F[1], N[1], PMat = P)
    replica_weight += tmp[1]
    p_sign *= tmp[2]
    tmp = compute_PF(F[2], N[2], PMat = P)
    replica_weight += tmp[1]
    p_sign *= tmp[2]

    p = replica_weight - sum(walker.weight)
    p_sign *= prod(walker.sign)

    return min(0, p), p_sign
end

function measure_TransitProb(system_ext::System, qmc_ext::QMC, walker1::Walker, walker2::Walker)
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2}
    """
    N = system_ext.N
    auxfield = hcat(walker1.auxfield, walker2.auxfield)
    P = walker1.tempdata.P

    ws = walker1.ws

    connected_weight = 0.0
    p_sign = 1.0

    F, dummy1, dummy2 = run_full_propagation(auxfield, system_ext, qmc_ext, ws)
    tmp = compute_PF(F[1], N[1], PMat = P)
    connected_weight += tmp[1]
    p_sign *= tmp[2]
    tmp = compute_PF(F[2], N[2], PMat = P)
    connected_weight += tmp[1]
    p_sign *= tmp[2]

    p = connected_weight - sum(walker1.weight) - sum(walker2.weight)
    p_sign *= prod(walker1.sign) * prod(walker2.sign)

    return min(0, p), p_sign
end

### GCE Measurements ###
function measure_TransitProb(
    system_og::System, qmc_og::QMC, walker::GCWalker; 
    G::AbstractMatrix{T} = similar(walker.G[1])
) where {T<:Number}
    """
        Measure the transition probability p_{1 -> 2} = min{1, W2/W1}
    """
    Lhalf = system_og.L
    expβμ = exp(system_og.β * system_og.μ)
    
    replica_weight = 0.0
    p_sign = 1.0

    @views F, dummy1, dummy2 = run_full_propagation(walker.auxfield[:, 1 : Lhalf], system_og, qmc_og, walker.ws)
    tmp = inv_IpμA!(G, F[1], expβμ, walker.ws) 
    replica_weight += -tmp[1]
    p_sign *= tmp[2]
    tmp =  inv_IpμA!(G, F[2], expβμ, walker.ws) 
    replica_weight += -tmp[1]
    p_sign *= tmp[2]

    @views F, dummy1, dummy2 = run_full_propagation(walker.auxfield[:, Lhalf + 1 : end], system_og, qmc_og, walker.ws)
    tmp = inv_IpμA!(G, F[1], expβμ, walker.ws) 
    replica_weight += -tmp[1]
    p_sign *= tmp[2]
    tmp =  inv_IpμA!(G, F[2], expβμ, walker.ws) 
    replica_weight += -tmp[1]
    p_sign *= tmp[2]

    p = replica_weight - sum(walker.weight)
    p_sign *= prod(walker.sign)

    return min(0, p), p_sign
end

function measure_TransitProb(
    system_ext::System, qmc_ext::QMC, walker1::GCWalker, walker2::GCWalker;
    G::AbstractMatrix{T} = similar(walker1.G[1])
) where {T<:Number}
    """
        Measure the transition probability p_{2 -> 1} = min{1, W1/W2}
    """
    auxfield = hcat(walker1.auxfield, walker2.auxfield)

    connected_weight = 0.0
    p_sign = 1.0
    expβμ = exp(system_ext.β * system_ext.μ)

    F, dummy1, dummy2 = run_full_propagation(auxfield, system_ext, qmc_ext, walker1.ws)
    tmp = inv_IpμA!(G, F[1], expβμ, walker1.ws)  
    connected_weight += -tmp[1]
    p_sign *= tmp[2]
    tmp = inv_IpμA!(G, F[2], expβμ, walker1.ws) 
    connected_weight += -tmp[1]
    p_sign *= tmp[2]

    p = connected_weight - sum(walker1.weight) - sum(walker2.weight)
    p_sign *= prod(walker1.sign) * prod(walker2.sign)

    return min(0, p), p_sign
end

### Fidelity Measurement ###
function measure_TransitProb(system::System, μ::Float64, walker::Walker)
    """
        Measure the transition probability p_{N -> μ}(β/2) = min{1, Z_{μ}(β) / exp(βμN)*Z_{N}(β)}
    """
    N = system.N
    F = walker.F

    expβμ = exp(system.β * μ)
    βμN = system.β * μ * sum(N)

    gc_weight = 0
    p_sign = 1.0

    tmp = compute_PF(F[1], expβμ)
    gc_weight += tmp[1]
    p_sign *= tmp[2]

    tmp = compute_PF(F[2], expβμ)
    gc_weight += tmp[1]
    p_sign *= tmp[2]

    p = gc_weight - sum(walker.weight) - βμN
    p_sign *= prod(walker.sign)

    return min(0, p), p_sign
end

function measure_TransitProb(system::System, qmc::QMC, μ::Float64, walker::GCWalker) 
    """
        Measure the transition probability p_{μ -> N}(β/2) = min{1, exp(βμN)*Z_{N}(β) / Z_{μ}(β)}
    """
    N = system.N
    ϵ = qmc.lrThld

    βμN = system.β * μ * sum(N)

    # Canonical Ensemble weight & sign
    c_weight = 0.0
    p_sign = 1.0

    udtlr = UDTlr(walker.F[1], N[1], ϵ)
    tmp = compute_PF(udtlr, N[1])
    c_weight += tmp[1]
    p_sign *= tmp[2]

    udtlr = UDTlr(walker.F[2], N[2], ϵ)
    tmp = compute_PF(udtlr, N[2])
    c_weight += tmp[1]
    p_sign *= tmp[2]

    p = c_weight + βμN - sum(walker.weight)
    p_sign *= prod(walker.sign)

    return min(0, real(p)), p_sign
end
