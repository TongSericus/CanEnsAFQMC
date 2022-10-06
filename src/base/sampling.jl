struct RegSample{T<:FloatType, Te, Ns}
    sgn::T
    nk_up::Vector{ComplexF64}
    nk_dn::Vector{ComplexF64}
    Ek::Te
    Ep::Te
    Etot::Te
end
