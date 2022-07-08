struct RegSample{T<:FloatType}
    sgn::T
    Ek::T
    Ep::T
    Etot::T
end

struct GCESample{T<:FloatType}
    sgn::T
    N::T
    Ek::T
    Ep::T
    Etot::T
end

Base.@kwdef struct EtgSample{T<:FloatType, Nu, Nd}
    sgn::Vector{T} = []
    expS2::Vector{T} = []
    expS2n_up::Vector{SizedVector{Nu, T, Vector{T}}} = []
    expS2n_dn::Vector{SizedVector{Nd, T, Vector{T}}} = []
end
