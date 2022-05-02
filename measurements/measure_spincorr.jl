"""
    Measure spin-spin correlation functions
"""
function add_vector(system::System,
    rx::Int64, ry::Int64, lx::Int64, ly::Int64)
    """
    Calculate (rx + lx, ry + ly) with PBC
    """
    return (rx + lx - 1) % system.Ns[1] + 1, (ry + ly - 1) % system.Ns[2] + 1
end

function spin_corr(
    system::System, G::Tuple{Matrix{T1}, Matrix{T2}}, 
    r1::Tuple{Int64,Int64}, r2::Tuple{Int64,Int64}
    ) where {T1<:FloatType, T2<:FloatType}
    """
    calculate spin correlation between two points
    """
    # convert to matrix indices
    d1 = (r1[2] - 1) * system.Ns[1] + r1[1]
    d2 = (r2[2] - 1) * system.Ns[1] + r2[1]
    return (
        wick2_converter(d2, d2, d1, d1, G[1]) -
        G[1][d2, d2] * G[2][d1, d1] -
        G[2][d2, d2] * G[1][d1, d1] +
        wick2_converter(d2, d2, d1, d1, G[2])
        )
end

function measure_spincorr_func(
    system::System, G::Tuple{Matrix{T1}, Matrix{T2}}, 
    path::Array{Tuple{Int64,Int64},1}
    ) where {T1<:FloatType, T2<:FloatType}
    """
    Measure the spin-spin correlation function along a given path

    # Argument
    path -> a triangular path in the lattice. For instance,
        [(0,0), (1,0), (2,0), (2, 1), (2, 2), (1, 1)] traces out
        such a path
    """
    Css = @MVector zeros(ComplexF64, length(path))
    for r1x = 1 : system.Ns[1]
        for r1y = 1 : system.Ns[2]
            for (i, l) in enumerate(path)
                r2 = add_vector(system, r1x, r1y, l[1], l[2])
                # spin correlation between r1 and r2
                Css[i] += spin_corr(system, G, (r1x, r1y), r2) / system.V
            end
        end
    end

    return real(Css)

end