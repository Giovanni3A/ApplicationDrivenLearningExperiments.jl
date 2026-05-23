# Hybrid storage entrypoint: runs LS, Opt (gradient via MPI), and eval for all three model_types.
# Run with: julia --project main.jl

using Pkg
cur_dir = @__DIR__
Pkg.activate(dirname(dirname(cur_dir)))

import JobQueueMPI

JQM = JobQueueMPI

const MODEL_TYPES = (:type, :many, :uni)
const CONFIG_FILE = joinpath(@__DIR__, "config.jl")
const N_PROCS = 12

function set_config_model_type(mt::Symbol)
    open(CONFIG_FILE, "r+") do f
        content = read(f, String)
        new_content = replace(
            content,
            r"^model_type = :\w+"m => "model_type = :$mt",
        )
        truncate(f, 0)
        seek(f, 0)
        write(f, new_content)
    end
end

# LS + Opt for each model type
for mt in MODEL_TYPES
    println("\n========== LS + Opt for model_type = $mt ==========")
    set_config_model_type(mt)
    include("ls.jl")
    JQM.mpiexec(exe -> run(`$exe -n $(N_PROCS) $(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "opt.jl"))`))
end

# Eval for each model type
for mt in MODEL_TYPES
    println("\n========== Eval for model_type = $mt ==========")
    set_config_model_type(mt)
    include("eval.jl")
end
