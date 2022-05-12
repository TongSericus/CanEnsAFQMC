Base.@kwdef struct MCSample
    sgn::Vector{ComplexF64} = ComplexF64[]
    Ek::Vector{ComplexF64} = ComplexF64[]
    Ep::Vector{ComplexF64} = ComplexF64[]
    Etot::Vector{ComplexF64} = ComplexF64[]
    nk::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
    Css::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
    expS2::Vector{ComplexF64} = ComplexF64[]
    expS2n::Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[]
end
