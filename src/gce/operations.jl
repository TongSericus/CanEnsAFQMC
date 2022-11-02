### Old Scheme ###
compute_PF(F::UDT, expβμ::Float64) = sum(log.(complex.(1 .+ expβμ*eigvals(F))))
compute_PF(F::UDTlr, expβμ::Float64) = let
    λ = eigvals(Diagonal(F.D) * F.T * F.U, sortby=abs)
    logZ = sum(log.(complex.(1 .+ expβμ*λ)))
    real(logZ), real(exp(imag(logZ)im))
end

function run_full_propagation_reverse(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T,E}; 
    K = qmc.K, stab_interval = qmc.stab_interval, 
    K_interval = qmc.K_interval,
    Ns = system.V,
    B = [Matrix{Float64}(I, Ns, Ns), Matrix{Float64}(I, Ns, Ns)],
    FC = Cluster(B = ldrs(B[1], 2 * K))
) where {T, E}
    """
    Propagate the full space-time lattice in the reverse order
    """
    MP = Cluster(Ns, 2 * K)

    F = ldrs(B[1], 2)

    for i in K : -1 : 1

        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * stab_interval + j]
            singlestep_matrix!(B[1], B[2], σ, system, tmpmat = ws.M)
            MP.B[i] = B[1] * MP.B[i]            # spin-up
            MP.B[K + i] = B[2] * MP.B[K + i]    # spin-down
        end

        copyto!(FC.B[i], F[1])
        copyto!(FC.B[K + i], F[2])

        rmul!(F[1], MP.B[i], ws)
        rmul!(F[2], MP.B[K + i], ws)
    end

    return F, MP, FC
end
