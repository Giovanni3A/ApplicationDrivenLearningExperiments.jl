# Analyze results.

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

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/model.jl")

## uni
model_type = 1
pretrain = false
include("utils/pretrain.jl")

# LS
model_state_load = JLD2.load(pretrained_model_state, "state")
Flux.loadmodel!(nn, model_state_load)
pred_model = ApplicationDrivenLearning.PredictiveModel(nn)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

ls_pred = deflatten(model.forecast(X_test')')
ls_G_err = ls_pred[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
ls_G_rmse = mean(ls_G_err .^ 2) .^ 0.5
ls_Pi_err = ls_pred[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
ls_Pi_rmse = mean(ls_Pi_err .^ 2) .^ 0.5
ls_cost = ADL.compute_cost(model, X_test, Y_dict_test)

println("Pretrained predictor test RMSE: G = $ls_G_rmse, Pi = $ls_Pi_rmse")
println("Pretrained predictor test Cost: $ls_cost")

# ADL
model_state_load = JLD2.load(final_model_state, "state")
Flux.loadmodel!(nn, model_state_load[1])
pred_model = ApplicationDrivenLearning.PredictiveModel(nn)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

gd_pred = deflatten(model.forecast(X_test')')
gd_G_err = gd_pred[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
gd_G_rmse = mean(gd_G_err .^ 2) .^ 0.5
gd_Pi_err = gd_pred[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
gd_Pi_rmse = mean(gd_Pi_err .^ 2) .^ 0.5
gd_cost = ADL.compute_cost(model, X_test, Y_dict_test)

println("Gradient-mode predictor test RMSE: G = $gd_G_rmse, Pi = $gd_Pi_rmse")
println("Gradient-mode predictor test Cost: $gd_cost")

# Visualize forecasts vs actuals for a single test sample

t = 1

t_y = deflatten(Y_test_clean)[t, :, :, :]
g = t_y[:, 1, :]
pi = t_y[:, 2, :]

ls_ghat = ls_pred[t, :, 1, :]
ls_pihat = ls_pred[t, :, 2, :]
gd_ghat = gd_pred[t, :, 1, :]
gd_pihat = gd_pred[t, :, 2, :]

import Plots

# 3 x 2 grid with generation and actuals vs forecast for all three projects
fig_1_1 = Plots.plot(g[:, 1], label = "Actual G", title = "Project 1", xlabel = "Hour", ylabel = "MW")
Plots.plot!(ls_ghat[:, 1], label = "LS G", linestyle = :dash)
Plots.plot!(gd_ghat[:, 1], label = "GD G", linestyle = :dashdot)

fig_1_2 = Plots.plot(pi[:, 1], label = "Actual Pi", title = "Project 1", xlabel = "Hour", ylabel = "\$/MWh")
Plots.plot!(ls_pihat[:, 1], label = "LS Pi", linestyle = :dash)
Plots.plot!(gd_pihat[:, 1], label = "GD Pi", linestyle = :dashdot)

fig_2_1 = Plots.plot(g[:, 2], label = "Actual G", title = "Project 2", xlabel = "Hour", ylabel = "MW")
Plots.plot!(ls_ghat[:, 2], label = "LS G", linestyle = :dash)
Plots.plot!(gd_ghat[:, 2], label = "GD G", linestyle = :dashdot)

fig_2_2 = Plots.plot(pi[:, 2], label = "Actual Pi", title = "Project 2", xlabel = "Hour", ylabel = "\$/MWh")
Plots.plot!(ls_pihat[:, 2], label = "LS Pi", linestyle = :dash)
Plots.plot!(gd_pihat[:, 2], label = "GD Pi", linestyle = :dashdot)

fig_3_1 = Plots.plot(g[:, 3], label = "Actual G", title = "Project 3", xlabel = "Hour", ylabel = "MW")
Plots.plot!(ls_ghat[:, 3], label = "LS G", linestyle = :dash)
Plots.plot!(gd_ghat[:, 3], label = "GD G", linestyle = :dashdot)

fig_3_2 = Plots.plot(pi[:, 3], label = "Actual Pi", title = "Project 3", xlabel = "Hour", ylabel = "\$/MWh")
Plots.plot!(ls_pihat[:, 3], label = "LS Pi", linestyle = :dash)
Plots.plot!(gd_pihat[:, 3], label = "GD Pi", linestyle = :dashdot)

final_fig = Plots.plot(
    fig_1_1,
    fig_1_2,
    fig_2_1,
    fig_2_2,
    fig_3_1,
    fig_3_2;
    layout = (3, 2),
    size = (1200, 800),
    title = "Forecasts vs Actuals for Test Sample t=$t",
)