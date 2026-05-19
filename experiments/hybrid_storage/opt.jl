# Hybrid storage optimization run with MPI.
# JQM.mpiexec(exe -> run(`$exe -n 12 $(Base.julia_cmd()) --project opt.jl`))

using Pkg
cur_dir = @__DIR__
Pkg.activate(dirname(dirname(cur_dir)))

using JuMP
using Statistics
using Random
import Gurobi
using Flux
import NNlib
import JLD2
import JobQueueMPI

using ApplicationDrivenLearning

JQM = JobQueueMPI

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/pretrain.jl")
include("utils/model.jl")

pred_model = ApplicationDrivenLearning.PredictiveModel(nn)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

time1 = time()
sol = ApplicationDrivenLearning.train!(
    model,
    X_train,
    Y_dict_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMPIMode;
        rule = Flux.Adam(LEARNING_RATE),
        epochs = N_EPOCHS,
        compute_cost_every = COMPUTE_EVERY,
        batch_size = BATCH_SIZE,
        time_limit = TIME_LIMIT,
        mpi_finalize = false
    ),
)

if JQM.is_controller_process()
    println("GradientMode training time: $(time() - time1)")

    gd_pred = deflatten(model.forecast(X_test')')
    gd_G_err = gd_pred[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
    gd_G_rmse = mean(gd_G_err .^ 2) .^ 0.5
    gd_Pi_err = gd_pred[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
    gd_Pi_rmse = mean(gd_Pi_err .^ 2) .^ 0.5
    gd_cost = ADL.compute_cost(model, X_test, Y_dict_test)

    println("Gradient-mode predictor test RMSE: G = $gd_G_rmse, Pi = $gd_Pi_rmse")
    println("Gradient-mode predictor test Cost: $gd_cost")

    JLD2.jldsave(final_model_state; state = Flux.state(model.forecast.networks))
    println("Final state saved to $final_model_state")
end
