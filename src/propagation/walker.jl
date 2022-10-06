Base.@kwdef struct Cluster{T}
    B::Vector{T}
end
Base.prod(C::Cluster{T}, a::UnitRange{Int64}) where T = @views prod(C.B[a])
Base.prod(C::Cluster{T}, a::StepRange{Int64, Int64}) where T = @views prod(C.B[a])
Cluster(Ns::Int64, N::Int64) = Cluster(B = [Matrix(1.0I, Ns, Ns) for _ in 1 : N])

struct TempData{Tf<:Number, Tp<:Number, F<:Factorization{Tf}, C}
    """
        Preallocated data
        FL/FR = (Q, D, T) -> matrix decompositions of the walker are stored and updated as the propagation goes
        FM -> merge of FL and FR
        P -> Poisson-binomial matrix
    """
    FL::Vector{F}
    FR::Vector{F}
    FM::Vector{F}
    P::Matrix{Tp}
    cluster::Cluster{C}
end

struct Walker{Tw<:Number, Tf<:Number, F<:Factorization{Tf}, Tp<:Number, C}
    """
        All the MC information stored in a single walker

        weight -> weight of the walker (spin-up/down portions are stored separately)
        auxfield -> configurations of the walker
        cluster -> matrix multiplication is also stored
    """
    weight::Vector{Tw}
    auxfield::Matrix{Int64}
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

    (L % qmc.stab_interval == 0) || @error "# of time slices should be divisible by the stablization interval"

    FL, cluster = run_full_propagation(auxfield, system, qmc, K = div(L, qmc.stab_interval))
    if qmc.isLowrank
        FR = [UDTlr(Ns), UDTlr(Ns)]
    elseif qmc.isCP
        FR = [UDT(Ns), UDT(Ns)]
    else
        FR = [UDR(Ns), UDR(Ns)]
    end

    tempdata = TempData(FL, FR, deepcopy(FR), zeros(ComplexF64, system.V+1, system.V), Cluster(system.V, 2 * k))

    weight = [calc_pf(FL[1], system.N[1], PMat=tempdata.P), calc_pf(FL[2], system.N[2], PMat=tempdata.P)]

    return Walker(weight, auxfield, tempdata, cluster)
end
