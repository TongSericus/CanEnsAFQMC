Base.@kwdef struct Cluster{T}
    B::Vector{T}
end

Base.prod(C::Cluster{T}, a::Vector{Int64}) where T = @views prod(C.B[a])

Cluster(Ns::Int64, N::Int64) = Cluster(B = [Matrix(1.0I, Ns, Ns) for _ in 1 : N])
Cluster(A::Factorization{Tf}, N::Int64) where Tf = Cluster(B = [similar(A) for _ in 1 : N])

struct TempData{Tf<:Number, Tp<:Number, F<:Factorization{Tf}, C}
    """
        Preallocated data
        
        FC -> list of all partial factorizations
        Fτ = (Q, D, T) -> matrix decompositions of the walker are stored and updated as the propagation goes
        FM -> merge of Ft and FC[t]
        P -> Poisson-binomial matrix
    """
    FC::Cluster{F}
    Fτ::Vector{F}
    FM::Vector{F}
    P::Matrix{Tp}
    cluster::Cluster{C}
end

struct Walker{Tw<:Number, Tf<:Number, F<:Factorization{Tf}, Tp<:Number, C}
    """
        All the MC information stored in a single walker

        weight -> weight of the walker (spin-up/down portions are stored separately)
        auxfield -> configurations of the walker
        F -> matrix factorizations
        cluster -> matrix multiplication is also stored
    """
    weight::Vector{Tw}
    auxfield::Matrix{Int64}
    F::Vector{F}
    tempdata::TempData{Tf, Tp, F, C}
    cluster::Cluster{C}
end

function Walker(system::System, qmc::QMC; auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1)
    """
        Initialize a walker with a random configuration
    """
    L = system.L
    Ns = system.V
    k = qmc.stab_interval

    F, cluster, Fcluster = run_full_propagation(auxfield, system, qmc)
    if qmc.isLowrank
        Fτ = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        Fτ = [UDT(Ns), UDT(Ns)]
    else
        Fτ = [UDR(Ns), UDR(Ns)]
    end

    tempdata = TempData(
        Fcluster, 
        Fτ, [similar(Fτ[1]), similar(Fτ[2])], 
        zeros(ComplexF64, system.V+1, system.V), 
        Cluster(Ns, 2 * k)
    )

    weight = [calc_pf(F[1], system.N[1], PMat=tempdata.P), calc_pf(F[2], system.N[2], PMat=tempdata.P)]

    return Walker(weight, auxfield, F, tempdata, cluster)
end
