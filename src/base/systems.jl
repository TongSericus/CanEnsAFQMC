"""
    Define the regular systems
"""

abstract type System end
abstract type Hubbard <: System end

"""
    Parameters for generic Hubbard models
        
    Ns -> number of sites in each dimension
    V -> volume of the lattce
    N -> number of spin-ups/downs
    T -> one-body, kinetic matrix
    U -> repulsion constant
    μ -> chemical potential used for the GCE calculations
    β -> inverse temperature
    L -> β / Δτ
    auxfield -> discrete HS variables sorted by field variables (±1) and spins (up/down),
                for instance, auxfield[2][1] represents spin-up section with σ = -1
    Bk -> exponential of the kinetic matrix
"""
struct GenericHubbard{T, Tk} <: Hubbard
    ### Model Constants ###
    Ns::Tuple{Int64, Int64, Int64}  # 3D
    V::Int64    # total dimension
    N::Tuple{Int64, Int64}  # spin-up and spin-dn
    T::Tk   # kinetic matrix (can be various forms)
    U::Float64
    Aidx::Vector{Int}

    ### Temperature and Chemical Potential ###
    μ::Float64
    β::Float64
    L::Int64
    
    ### Automatically-generated Constants ###
    useChargeHST::Bool
    auxfield::Vector{T}
    V₊::Vector{T}
    V₋::Vector{T}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Tk
    Bk⁻¹::Tk

    function GenericHubbard(
        Ns::Tuple{Int64, Int64, Int64}, N::Tuple{Int64, Int64},
        T::AbstractMatrix, U::Float64,
        μ::Float64, β::Float64, L::Int64;
        sys_type::DataType = ComplexF64,
        Aidx::Vector{Int} = [1],
        useChargeHST::Bool = false,
        useFirstOrderTrotter::Bool = false
    )
        Δτ = β / L
        useFirstOrderTrotter ? dτ = Δτ : dτ = Δτ/2
        Bk = exp(-T * dτ)
        Bk⁻¹ = exp(T * dτ)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useChargeHST
            γ = sys_type == ComplexF64 ? acosh(exp(-Δτ * U / 2) + 0im) : acosh(exp(-Δτ * U / 2))
            # use symmetric Hubbard potential
            auxfield = [exp(γ), exp(-γ)]
        else
            γ = sys_type == ComplexF64 ? acosh(exp(Δτ * U / 2) + 0im) : acosh(exp(Δτ * U / 2))
            auxfield = [exp(γ), exp(-γ)]
        end

        # add chemical potential
        @. auxfield *= exp.(μ * Δτ)

        V = prod(Ns)
        V₊ = zeros(sys_type, V)
        V₋ = zeros(sys_type, V)

        return new{sys_type, typeof(Bk)}(
            Ns, V, 
            N, T, U,
            Aidx,
            μ, β, L,
            useChargeHST, auxfield, V₊, V₋,
            useFirstOrderTrotter,
            Bk, Bk⁻¹
        )
    end
end
