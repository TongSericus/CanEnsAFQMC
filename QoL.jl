"""
    Some quality-of-life functions
"""

function pinpoint_Fermilevel(expβϵ::Vector{T}, N::Int64) where {T<:FloatType}
    Ns = length(expβϵ)
    return (abs(expβϵ[Ns - N + 1]) + abs(expβϵ[Ns - N])) / 2
end