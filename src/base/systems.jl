abstract type System end

mutable struct Hubbard <: System
    """
        Constants in the simulation
        
        Ns -> number of sites in each dimension
        V -> volume of the lattce
        N[1] -> number of spin-ups
        N[2] -> number of spin-downs
        t -> hopping constant
        U -> repulsion constant
        T -> one-body, kinetic matrix (used for one-body measurements)
        μ -> chemical potential used for the GCE calculations
        Δτ -> imaginary time interval
        L -> β / Δτ
        auxfield -> discrete HS variables sorted by field variables (±1) and spins (up/down),
                    for instance, auxfield[2][1] represents spin-up section with σ = -1
        Bk -> exponential of the kinetic matrix
        BT -> trial propagator matrix
        BTinv -> inverse of trial propagator matrix
    """
    ### Model Constants ###
    Ns::Tuple{Int64, Int64}
    V::Int64
    N::Tuple{Int64, Int64}
    t::Float64
    U::Float64
    T::Array{Float64,2}
    μ::Float64
    expβμ::Float64
    expiφ::Vector{ComplexF64}
    ### AFQMC Constants ###
    Δτ::Float64
    L::Int64
    ### Automatically-Generated Constants ###
    auxfield::Vector{Vector{Float64}}
    Bk::Matrix{Float64}
    BT::Matrix{Float64}
    BTinv::Matrix{Float64}

    function Hubbard(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, U::Float64,
        μ::Float64, Δτ::Float64, L::Int64
    )
        if Ns[2] == 1 
            T = kinetic_matrix_hubbard1D(Ns[1], t)
        else
            T = kinetic_matrix_hubbard2D(Ns[1], Ns[2], t)
        end
        γ = atanh(sqrt(tanh(Δτ * U / 4)))
        auxfield = [
            [exp(2 * γ - Δτ * U / 2), exp(-2 * γ - Δτ * U / 2)],
            [exp(-2 * γ - Δτ * U / 2), exp(2 * γ - Δτ * U / 2)]
        ]
        expiφ = exp.(im * [2 * π * m / (prod(Ns) + 1) for m = 1 : prod(Ns) + 1])

        return new(
            Ns, prod(Ns), N, t, U, T, 
            μ, exp(Δτ * L * μ), expiφ, 
            Δτ, L, auxfield,
            exp(-T * Δτ/2), exp(-T * Δτ), inv(exp(-T * Δτ))
        )

    end

end