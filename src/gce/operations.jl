### Old Scheme ###
calc_pf(F::UDT, expβμ::Float64) = sum(log.(complex.(1 .+ expβμ*eigvals(F))))

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

### A New Scheme using StableLinearAlgebra Package ###
function run_full_propagation(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T}; 
    K = qmc.K, stab_interval = qmc.stab_interval, K_interval = qmc.K_interval
) where {T<:Number}
    Ns = system.V

    B = [Matrix{Float64}(undef, Ns, Ns), Matrix{Float64}(undef, Ns, Ns)]
    MP = Cluster(Ns, 2 * K)

    F = ldrs(B[1], 2)
    FC = Cluster(B = ldrs(B[1], 2 * K))

    for i in 1 : K

        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * stab_interval + j]
            singlestep_matrix!(B[1], B[2], σ, system)
            MP.B[i] = B[1] * MP.B[i]            # spin-up
            MP.B[K + i] = B[2] * MP.B[K + i]    # spin-down
        end

        copyto!(FC.B[i], F[1])
        copyto!(FC.B[K + i], F[2])

        lmul!(MP.B[i], F[1], ws)
        lmul!(MP.B[K + i], F[2], ws)
    end

    return F, MP, FC
end
