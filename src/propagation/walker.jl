struct Walker{T1<:FloatType, T2<:FloatType}
    """
        All the MC information carried by a single walker

        weight -> weight of the walker (spin-up/down portions are stored separately)
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        F = (Q, D, T) -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Vector{T1}
    auxfield::Matrix{Int64}
    F::Vector{UDT{T2}}
end

function Walker(system::System, qmc::QMC)
    """
    Initialize a walker with a random configuration
    """
    # initialize a random field configuration
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1
    F = full_propagation(auxfield, system, qmc)
    # diagonalize the decomposition
    expβϵ = [eigvals(F[1]), eigvals(F[2])]
    # calculate the statistical weight
    Z = [
        pf_projection(system.V, system.N[1], expβϵ[1], system.expiφ, false),
        pf_projection(system.V, system.N[2], expβϵ[2], system.expiφ, false)
    ]

    Walker(Z, auxfield, F)
end

struct ConstrainedWalker{T1<:FloatType, T2<:FloatType}
    """
        All the sampling information carried by a single walker

        weight -> weight of the walker
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        Q, D, T -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Vector{T1}
    Pl::Vector{Float64}
    auxfield::Array{Int64,2}
    F::Vector{UDT{T2}}
end

function ConstrainedWalker(system::System, qmc::QMC)
    """
    Initialize a constrained walker with a random configuration
    """
    # initialize a random field configuration
    auxfield = zeros(Int64, system.V, system.L)
    F = full_propagation(auxfield, system, qmc)
    # diagonalize the decomposition
    expβϵ = [eigvals(F[1]), eigvals(F[2])]
    # calculate the statistical weight
    Z = [
        pf_projection(system.V, system.N[1], expβϵ[1], system.expiφ, false),
        pf_projection(system.V, system.N[2], expβϵ[2], system.expiφ, false)
    ]

    # construct the walker
    ConstrainedWalker(Z, [1.0, 1.0], auxfield, F)
end

init_walker_list(system::System, qmc::QMC) = let 
    walker = ConstrainedWalker(system, qmc)
    [deepcopy(walker) for _ = 1 : qmc.ntotwalkers]
end
