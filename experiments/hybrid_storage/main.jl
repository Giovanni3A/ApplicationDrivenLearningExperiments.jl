# Hybrid storage experiment orchestration.
# Run: julia --project=. experiments/hybrid_storage/main.jl

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

using ApplicationDrivenLearning

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/pretrain.jl")

# Stand up the LP (skip during pretrain-only? matpower runs it anyway so the
# input_output_map and Y_dicts are available — we do the same). Wrap nns into a
# PredictiveModel and compute the LS-pretrain baseline cost.
include("utils/model.jl")

pred_model = ApplicationDrivenLearning.PredictiveModel(nn)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

# Baseline metrics from the pretrained predictor.
ls_pred = denormalize_and_deflatten(model.forecast(X_test')')

ls_G_err = ls_pred[:, :, 1, :] - deflattened_Y_test[:, :, 1, :]
G_rmse = mean(ls_G_err .^ 2) .^ 0.5

ls_Pi_err = ls_pred[:, :, 2, :] - deflattened_Y_test[:, :, 2, :]
Pi_rmse = mean(ls_Pi_err .^ 2) .^ 0.5

println("Pretrained predictor test RMSE: G = $G_rmse, Pi = $Pi_rmse")

println("Running gradient-mode decision-focused training...")
time1 = time()
sol = ApplicationDrivenLearning.train!(
    model,
    X_train,
    Y_dict_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode;
        rule = Flux.Adam(LEARNING_RATE),
        epochs = N_EPOCHS,
        compute_cost_every = COMPUTE_EVERY,
        batch_size = BATCH_SIZE,
        time_limit = TIME_LIMIT,
    ),
)
println("GradientMode training time: $(time() - time1)")

gd_pred = model.forecast(X_train')'
gd_mse = mean(sum((gd_pred .- Y_train) .^ 2, dims = 2))
gd_cost = ApplicationDrivenLearning.compute_cost(model, X_train, Y_dict_train)
println("Gradient-trained $(ARCH_NAME) MSE (train):  $gd_mse")
println("Gradient-trained $(ARCH_NAME) Cost (train): $gd_cost")

JLD2.jldsave(final_model_state; state = Flux.state(model.forecast.networks))
println("Final state saved to $final_model_state")

#################################################################################

import ParametricOptInterface as POI
import MathOptInterface as MOI

yhat = model.forecast(X_train[1, :])
MOI.set.(model.plan, POI.ParameterValue(), model.plan_forecast_params, yhat)
JuMP.optimize!(ADL.Plan(model))

set_normalized_rhs.(
    model.assess[:assess_policy_fix],
    value.(ADL.plan_policy_vars(model)),
)
fix.(ADL.assess_forecast_vars(model), yhat; force = true)
JuMP.optimize!(ADL.Assess(model))

c = JuMP.value.(charge.assess)
d = JuMP.value.(discharge.assess)
s = JuMP.value.(assess_sale)
soc = JuMP.value.(assess_soc)
short = JuMP.value.(assess_shortfall)
curt = JuMP.value.(assess_curtail)

JuMP.objective_value(ADL.Assess(model))

sum(short) * PHANTOM_PENALTY

hat_norm = reshape(yhat, H, N_PROJ, 2)
ghat_norm = hat_norm[:, :, 1]
pihat_norm = hat_norm[:, :, 2]
ghat = [
    ghat_norm[h, pidx] * std_Y[y_idx_G(pidx, h)] + mu_Y[y_idx_G(pidx, h)]
    for h = 1:H, pidx = 1:N_PROJ
]
pihat = [
    pihat_norm[h, pidx] * std_Y[y_idx_pi(pidx, h)] + mu_Y[y_idx_pi(pidx, h)] for h = 1:H, pidx = 1:N_PROJ
]

gval_norm = [
    Y_dict_train[G_fc[h, pidx]][1] * std_Y[y_idx_G(pidx, h)] +
    mu_Y[y_idx_G(pidx, h)] for h = 1:H, pidx = 1:N_PROJ
]
pival_norm = [
    Y_dict_train[pi_fc[h, pidx]][1] * std_Y[y_idx_pi(pidx, h)] +
    mu_Y[y_idx_pi(pidx, h)] for h = 1:H, pidx = 1:N_PROJ
]

using Plots

proj = 3

oper_fig = plot(
    c[:, proj],
    label = "Charge",
    title = "Battery operation (project $proj)",
)
plot!(oper_fig, d[:, proj], label = "Discharge")
plot!(oper_fig, s[:, proj] - d[:, proj], label = "Sale from Renewable")
plot!(oper_fig, ghat[:, proj], label = "G_hat", color = :black)

# 2 by 2 plots grid (sized 30, 10)
plot(
    plot(pihat[:, proj], label = "pi_hat", title = "pi_hat"),
    oper_fig,
    plot(soc[:, proj], label = "SoC", title = "State of Charge"),
    layout = (3, 1),
    size = (800, 600),
)