# LS model training and evaluation script.

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

import ApplicationDrivenLearning

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/pretrain.jl")
include("utils/model.jl")

pred_model = ApplicationDrivenLearning.PredictiveModel(nn)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

# Baseline metrics from the pretrained predictor.
ls_pred = deflatten(model.forecast(X_test')')
ls_G_err = ls_pred[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
ls_G_rmse = mean(ls_G_err .^ 2) .^ 0.5
ls_Pi_err = ls_pred[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
ls_Pi_rmse = mean(ls_Pi_err .^ 2) .^ 0.5
ls_cost = ADL.compute_cost(model, X_test, Y_dict_test)

println("Pretrained predictor test RMSE: G = $ls_G_rmse, Pi = $ls_Pi_rmse")
println("Pretrained predictor test Cost: $ls_cost")


####################################### PLOTS ANALYSIS ##########################################

# import ParametricOptInterface as POI
# import MathOptInterface as MOI

# # run plan with forecasts
# yhat = model.forecast(X_train[43, :])
# MOI.set.(model.plan, POI.ParameterValue(), model.plan_forecast_params, yhat)
# JuMP.optimize!(ADL.Plan(model))

# # extract plan solution
# planned_charge = JuMP.value.(plan_charge)
# planned_discharge = JuMP.value.(plan_discharge)

# # run assessment with actuals and plan decisions
# set_normalized_rhs.(
#     model.assess[:assess_policy_fix],
#     value.(ADL.plan_policy_vars(model)),
# )
# y = ADL.dict_to_var_indexed_matrix(Y_dict_train, model.forecast_vars)[1, :]
# fix.(ADL.assess_forecast_vars(model), y; force = true)
# JuMP.optimize!(ADL.Assess(model))

# # extract assessment solution
# actual_charge = JuMP.value.(assess_charge)
# actual_discharge = JuMP.value.(assess_discharge)
# actual_sale = JuMP.value.(assess_sale)
# actual_soc = JuMP.value.(assess_soc)
# actual_curt = JuMP.value.(assess_curtail)
# actual_g_out = JuMP.value.(assess_generator_output)

# JuMP.objective_value(ADL.Assess(model))

# ghat = deflatten(yhat)[:, 1, :]
# pihat = deflatten(yhat)[:, 2, :]
# g = deflatten(Y_train_clean[1, :])[:, 1, :]
# pi = deflatten(Y_train_clean[1, :])[:, 2, :]

# using Plots

# proj = 1

# # price fig
# price_fig = plot(
#     pi[:, proj],
#     label = "Price",
#     title = "Price (project $proj)",
#     color=:black
# )
# plot!(price_fig, pihat[:, proj], label = "Pi_hat", color = :grey)

# # generation fig
# gen_fig = plot(
#     g[:, proj],
#     label = "Generation",
#     title = "Generation (project $proj)",
#     color=:black
# )
# plot!(gen_fig, ghat[:, proj], label = "G_hat", color = :grey)

# # charge fig
# charge_fig = plot(
#     planned_charge[:, proj],
#     label = "Charge",
#     title = "Battery charge (project $proj)",
#     color=:grey
# )
# plot!(charge_fig, actual_charge[:, proj], label = "Charge", color=:black)

# # discharge fig
# discharge_fig = plot(
#     planned_discharge[:, proj],
#     label = "Discharge",
#     title = "Battery discharge (project $proj)",
#     color=:grey
# )
# plot!(discharge_fig, actual_discharge[:, proj], label = "Discharge", color=:black)

# # sale fig
# sale_fig = plot(
#     actual_sale[:, proj],
#     label = "Sale",
#     title = "Grid sales (project $proj)",
#     color=:black
# )
# plot!(sale_fig, actual_curt[:, proj], label = "Curtailment", color=:red)

# # soc fig
# soc_fig = plot(
#     actual_soc[:, proj],
#     label = "SoC",
#     title = "State of Charge (project $proj)",
#     color=:black
# )

# # grid of figs
# plot(
#     price_fig,
#     gen_fig,
#     charge_fig,
#     discharge_fig,
#     sale_fig,
#     soc_fig,
#     layout = (6, 1),
#     size = (600, 1000),
#     legend = nothing,
# )