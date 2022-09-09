Base.@kwdef struct Cluster{T}
    B::Vector{T}
end
Base.prod(C::Cluster{T}, a::UnitRange{Int64}) where T = prod(C.B[a])
Base.prod(C::Cluster{T}, a::StepRange{Int64, Int64}) where T = prod(C.B[a])
Cluster(Ns::Int64, N::Int64) = Cluster(B = SizedMatrix{Ns, Ns}.([Matrix(1.0I, Ns, Ns) for _ in 1 : N]))

struct Walker{T1<:FloatType, T2<:FloatType, F<:Factorization{T2}, T}
    """
        All the MC information carried by a single walker

        weight -> weight of the walker (spin-up/down portions are stored separately)
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        F = (Q, D, T) -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Vector{T1}
    auxfield::Matrix{Int64}
    F::Vector{F}
    cluster::Cluster{T}
end

function Walker(system::System, qmc::QMC; auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1)
    """
    Initialize a walker with a random configuration
    """
    L = size(auxfield)[2]
    (L % qmc.stab_interval == 0) || @error "# of time slices should be divisible by the stablization interval"
    F, cluster = initial_propagation(auxfield, system, qmc, K = div(L, qmc.stab_interval))
    # diagonalize the decomposition
    λ = [eigvals(F[1]), eigvals(F[2])]
    # calculate the statistical weight
    logZ = [
        pf_recursion(length(λ[1]), system.N[1], λ[1]),
        pf_recursion(length(λ[2]), system.N[2], λ[2])
    ]

    return Walker{eltype(logZ), eltype(F[1].U), typeof(F[1]), eltype(cluster.B)}(logZ, auxfield, F, cluster)
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
        pf_recursion(system.V, system.N[1], expβϵ[1]),
        pf_recursion(system.V, system.N[2], expβϵ[2])
    ]

    # construct the walker
    system.isReal && return ConstrainedWalker{Float64, eltype(F[1].U)}(real(Z), [1.0, 1.0], auxfield, F)
    return ConstrainedWalker{ComplexF64, eltype(F[1].U)}(Z, [1.0, 1.0], auxfield, F)
end

init_walker_list(system::System, qmc::QMC) = let 
    walker = ConstrainedWalker(system, qmc)
    [deepcopy(walker) for _ = 1 : qmc.ntotwalkers]
end
