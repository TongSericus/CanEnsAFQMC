module CanEnsAFQMC

using Reexport: @reexport
@reexport using LinearAlgebra, Statistics, Random, StaticArrays

export FloatType
include("./base/constants.jl")

export QMC, GeneralMeasure, EtgMeasure
export System, Hubbard
include("./base/systems.jl")
include("./base/matrix_generator.jl")
include("./base/variable.jl")

export  fermilevel, poissbino, sum_antidiagonal,
    pf_recursion, occ_recursion,
    pf_projection, occ_projection,
    QRCP_update
include("./utils/quickmath.jl")
include("./utils/ce_recursion.jl")
include("./utils/ce_projection.jl")
include("./utils/linalg.jl")

export Walker, ConstrainedWalker, TrialWalker, MatDecomp, QDT
include("./propagation/walker.jl")
include("./propagation/operations.jl")
export sweep!, sweep!_replica
include("./propagation/metropolis.jl")
include("./propagation/constraint_path.jl")
include("./propagation/replica.jl")

export WalkerProfile,
    measure_energy_hubbard,
    measure_renyi2_entropy
include("./measurements/walkerprofile.jl")
include("./measurements/measure_energy.jl")
include("./measurements/measure_etgent.jl")

end
