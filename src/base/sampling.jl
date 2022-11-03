struct RegSample{T<:Number, Te}
    sgn::T
    nk_up::Vector{ComplexF64}
    nk_dn::Vector{ComplexF64}
    Ek::Te
    Ep::Te
    Etot::Te
end
