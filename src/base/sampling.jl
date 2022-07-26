struct RegSample{T<:FloatType, Ns}
    sgn::T
    G_up::SizedMatrix{Ns, Ns, ComplexF64}
    G_dn::SizedMatrix{Ns, Ns, ComplexF64}
end

Base.@kwdef struct EtgSample{T<:FloatType, Nu, Nd}
    sgn::Vector{T} = []
    expS2::Vector{T} = []
    expS2n_up::Vector{SizedVector{Nu, T, Vector{T}}} = []
    expS2n_dn::Vector{SizedVector{Nd, T, Vector{T}}} = []
end
