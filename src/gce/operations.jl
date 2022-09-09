function shiftB(F::UDT, B::AbstractMatrix)
    F = QR_rmul(F, inv(B))
    F = QR_lmul(B, F)
end

function computeG(F::UDT, expβμ::Float64; 
    r = similar(F.U), l = similar(F.U), Dp = similar(F.D), Dm = similar(F.D)
)
    """
        Stabilized calculation of G = [1 + UDT]^(-1)
    """
    U, D, T = F
    Dμ = D * expβμ
    Dp .= max.(Dμ, 1.)
    Dm .= min.(Dμ, 1.)

    Dp .\= 1
    Dpinv = Dp

    ldiv!(l, lu(T), Diagonal(Dpinv))
    mul!(r, U, Diagonal(Dm))
    r .+= l
    u, d, t = UDT(r)
    ldiv!(r, lu!(t), Diagonal(1 ./ d))
    mul!(l, r, u')

    lmul!(Diagonal(Dpinv), l)
    u, d, t = UDT(l)
    ldiv!(l, lu(T), u)

    return UDT(l, d, t)
end

function updateG(G::Matrix{T}, α::Float64, d::Float64, sidx::Int64) where {T<:FloatType}
    @views dG = α / d * (I - G)[:, sidx] * (G[sidx, :])'
    G = G - dG
end

function recomputeG(system::System, qmc::QMC, walker::GCEWalker, cidx::Int64) where T
    """
        Recompute G for calibration
    """
    expβμ = walker.expβμ
    cluster = walker.cluster

    F = partial_propagation(cluster, system, qmc, circshift(collect(1 : qmc.K), -cidx))

    F = [shiftB(F[1], system.Bk), shiftB(F[2], system.Bk)]
    F = [computeG(F[1], expβμ), computeG(F[2], expβμ)]
    G = [Matrix(F[1]), Matrix(F[2])]

    return GCEWalker{Float64, eltype(cluster.B)}(walker.α, expβμ, walker.auxfield, G, cluster)
end
