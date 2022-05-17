Base.@kwdef struct RegSample{T <: FloatType, N, Nc}
    sgn::Vector{T} = []
    Ek::Vector{T} = []
    Ep::Vector{T} = []
    Etot::Vector{T} = []
    nk::Vector{SizedVector{N, T, Vector{T}}} = []
    Css::Vector{SizedVector{Nc, T, Vector{T}}} = []
end

Base.@kwdef struct EtgSample{T<:FloatType, Nu, Nd}
    sgn::Vector{T} = []
    expS2::Vector{T} = []
    expS2n_up::Vector{SizedVector{Nu, T, Vector{T}}} = []
    expS2n_dn::Vector{SizedVector{Nd, T, Vector{T}}} = []
end
