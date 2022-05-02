mutable struct Walker{T1<:AbstractMatrix, T2<:AbstractMatrix}
    """
        All the MC information carried by a single walker

        weight -> weight of the walker (spin-up/down portions are stored separately)
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        Q, D, T -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Vector{Float64}
    sgn::ComplexF64
    auxfield::Matrix{Int64}
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
    update_count::Int64
end

function Walker(system::System, qmc::QMC)
    """
    Initialize a walker with a random configuration
    """
    # initialize a random field configuration
    σfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1
    Q, D, T = qmc.isLowrank ?
        full_propagation_lowrank(σfield, system, qmc) :
        full_propagation(σfield, system, qmc)
    # diagonalize the decomposition
    expβϵ = [
        eigvals(D[1] * T[1] * Q[1], sortby = abs),
        eigvals(D[2] * T[2] * Q[2], sortby = abs)
    ]
    # fill the truncated slots with some super small numbers
    expβϵ = [
        vcat(1e-10 * ones(system.V - length(expβϵ[1])), expβϵ[1]),
        vcat(1e-10 * ones(system.V - length(expβϵ[2])), expβϵ[2])
    ]
    # calculate the statistical weight
    Z = [
        pf_projection(system.V, system.N[1], expβϵ[1], system.expiφ, false),
        pf_projection(system.V, system.N[2], expβϵ[2], system.expiφ, false)
    ]

    # construct the walker
    return Walker(
        real(Z), sgn(Z[1] * Z[2]),
        σfield,
        Q, D, T,
        0
    )

end

mutable struct ConstrainedWalker{T1<:AbstractMatrix, T2<:AbstractMatrix}
    """
        All the sampling information carried by a single walker

        weight -> weight of the walker
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        Q, D, T -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Float64
    Pl::Vector{Float64}
    auxfield::Array{Int64,2}
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
    update_count::Int64
end

function ConstrainedWalker(system::System, qmc::QMC)
    """
    Initialize a constrained walker with a random configuration
    """
    # initialize a random field configuration
    σfield = zeros(Int64, system.V, system.L)
    Q, D, T = qmc.isLowrank ?
        full_propagation_lowrank(σfield, system, qmc) :
        full_propagation(σfield, system, qmc)
    # diagonalize the decomposition
    expβϵ = [
        eigvals(D[1] * T[1] * Q[1], sortby = abs),
        eigvals(D[2] * T[2] * Q[2], sortby = abs)
    ]
    # fill the truncated slots with some super small numbers
    expβϵ = [
        vcat(1e-10 * ones(system.V - length(expβϵ[1])), expβϵ[1]),
        vcat(1e-10 * ones(system.V - length(expβϵ[2])), expβϵ[2])
    ]
    # calculate the statistical weight
    Z = [
        pf_projection(system.V, system.N[1], expβϵ[1], false),
        pf_projection(system.V, system.N[2], expβϵ[2], false)
    ]

    # construct the walker
    return ConstrainedWalker(
        Z, [1.0, 1.0],
        σfield,
        Q, D, T,
        0
    )

end

struct TrialWalker{T1<:AbstractMatrix, T2<:AbstractMatrix, T3<:AbstractMatrix}
    """
        Walker information for each trial step
    """
    weight::Vector{Float64}
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T3}
end

abstract type MatDecomp end

mutable struct QDT{T1<:AbstractMatrix, T2<:AbstractMatrix} <: MatDecomp
    """
        Temporal QR decomposion
    """
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
end

init_walker_list(system::System, qmc::QMC) = let 
    walker = ConstrainedWalker(system, qmc)
    [deepcopy(walker) for _ = 1 : qmc.ntotwalkers]
end
