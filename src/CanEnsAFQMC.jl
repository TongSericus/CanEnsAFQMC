module CanEnsAFQMC

    using Reexport: @reexport
    @reexport using LinearAlgebra, StableLinearAlgebra, Random, FFTW

    export System, Hubbard, GenericHubbard
    include("./base/systems.jl")
    export hopping_matrix_Hubbard_1d, hopping_matrix_Hubbard_2d
    include("./base/matrix_generator.jl")
    export QMC
    include("./base/variable.jl")

    # math functions and canonical ensemble projections
    include("./utils/quickmath.jl")
    include("./utils/ce_recursion.jl")
    include("./utils/ce_projection.jl")

    export LDRLowRank
    include("./utils/linalg.jl")
    include("./utils/linalg_lowrank.jl")

    export Walker
    include("./propagation/walker.jl")
    include("./propagation/operations.jl")
    export sweep!, sweep!_asymmetric, sweep!_symmetric
    include("./propagation/metropolis.jl")

end
