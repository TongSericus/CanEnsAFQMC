abstract type MCSample end

Base.@kwdef struct MCSampleReal{Float64} <: MCSample
    sgn::Vector{Float64} = Float64[]
    Ek::Vector{Float64} = Float64[]
    Ep::Vector{Float64} = Float64[]
    Etot::Vector{Float64} = Float64[]
    nk::Vector{Vector{Float64}} = Vector{Float64}[]
    Css::Vector{Vector{Float64}} = Vector{Float64}[]
    expS2::Vector{Float64} = Float64[]
    expS2n_up::Vector{Vector{Float64}} = Vector{Float64}[]
    expS2n_dn::Vector{Vector{Float64}} = Vector{Float64}[]
end

Base.@kwdef struct MCSampleComplex{ComplexF64} <: MCSample
    sgn::Vector{ComplexF64} = ComplexF64[]
    Ek::Vector{ComplexF64} = ComplexF64[]
    Ep::Vector{ComplexF64} = ComplexF64[]
    Etot::Vector{ComplexF64} = ComplexF64[]
    nk::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
    Css::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
    expS2::Vector{ComplexF64} = Float64[]
    expS2n_up::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
    expS2n_dn::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
end
