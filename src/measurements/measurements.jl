struct Measurements{T}
    # Energy -> Ek, Ep, Etot
    E::Vector{T}

    # Heat Capacity
    Z_pmβ::Vector{T}
    H_pmβ::Vector{T}
    sign_pmβ::Vector{T}

    # Momentum Distribution
    nk::T

    # Charge Structural Factor
    Sq::T
end

function Measurements(system::System)
end