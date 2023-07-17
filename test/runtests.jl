using CanEnsAFQMC, Test

#######################
##### Test Module #####
#######################
@testset "CanEnsAFQMC" begin
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (8, 8),
        # t, U
        T, 4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type=Float64,
        # if use charge decomposition
        useChargeHST=false,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter=false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        nwarmups=512, nsamples=1024, measure_interval=6,
        # stablization interval
        stab_interval=10,
        # use low-rank approximation and set the threshold
        isLowrank=false, lrThld=1e-8,
        # debugging flag
        saveRatio=true
    )

    qmc_nosave = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        nwarmups=512, nsamples=1024, measure_interval=6,
        # stablization interval
        stab_interval=10,
        # use low-rank approximation and set the threshold
        isLowrank=false, lrThld=1e-8,
        # debugging flag
        saveRatio=false
    )

    walker = Walker(system, qmc)
    sweep!(system, qmc_nosave, walker, loop_number=5)

    ### test regular update ###

    ## forward sweep ##
    auxfield = copy(walker.auxfield)
    sweep!_asymmetric(system, qmc, walker, direction=1)
    # pick a random point in the space-time lattice
    idx = rand(1 : system.V*system.L)
    @. auxfield[1:idx-1] = walker.auxfield[1:idx-1]
    walker′ = Walker(system, qmc, auxfield=auxfield)
    auxfield[idx] *= -1
    walker″ = Walker(system, qmc, auxfield=auxfield)
    r = exp(sum(walker″.weight) - sum(walker′.weight))
    # and test if the correct Metropolis ratio is produced
    @test isapprox(r, walker.tmp_r[idx], atol=1e-5)

    ## backward sweep ##
    auxfield = copy(walker.auxfield)
    sweep!_asymmetric(system, qmc, walker, direction=2)
    # pick a random point in the space-time lattice
    idx_t = rand(1:system.L)
    idx_x = rand(1:system.V)
    idx = (system.L-idx_t)*system.V + idx_x + system.V*system.L
    @. auxfield[:, idx_t+1:system.L] = walker.auxfield[:, idx_t+1:system.L]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    walker′ = Walker(system, qmc, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = Walker(system, qmc, auxfield=auxfield)
    r = exp(sum(walker″.weight) - sum(walker′.weight))
    # and test if the correct Metropolis ratio is produced
    @test isapprox(r, walker.tmp_r[idx], atol=1e-5)

    ### test low-rank update ###
    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        nwarmups=512, nsamples=1024, measure_interval=6,
        # stablization interval
        stab_interval=10,
        # use low-rank approximation and set the threshold
        isLowrank=true, lrThld=1e-6,
        # debugging flag
        saveRatio=true
    )

    walker = Walker(system, qmc)
    sweep!(system, qmc_nosave, walker, loop_number=5)

    ### test regular update ###

    ## forward sweep ##
    auxfield = copy(walker.auxfield)
    sweep!_asymmetric(system, qmc, walker, direction=1)
    # pick a random point in the space-time lattice
    idx = rand(1 : system.V*system.L)
    @. auxfield[1:idx-1] = walker.auxfield[1:idx-1]
    walker′ = Walker(system, qmc, auxfield=auxfield)
    auxfield[idx] *= -1
    walker″ = Walker(system, qmc, auxfield=auxfield)
    r = exp(sum(walker″.weight) - sum(walker′.weight))
    # and test if the correct Metropolis ratio is produced
    @test isapprox(r, walker.tmp_r[idx], atol=1e-5)

    ## backward sweep ##
    auxfield = copy(walker.auxfield)
    sweep!_asymmetric(system, qmc, walker, direction=2)
    # pick a random point in the space-time lattice
    idx_t = rand(1:system.L)
    idx_x = rand(1:system.V)
    idx = (system.L-idx_t)*system.V + idx_x + system.V*system.L
    @. auxfield[:, idx_t+1:system.L] = walker.auxfield[:, idx_t+1:system.L]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    walker′ = Walker(system, qmc, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = Walker(system, qmc, auxfield=auxfield)
    r = exp(sum(walker″.weight) - sum(walker′.weight))
    # and test if the correct Metropolis ratio is produced
    @test isapprox(r, walker.tmp_r[idx], atol=1e-5)
end