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
pretrain = true
include("utils/pretrain.jl")
include("utils/model.jl")

ApplicationDrivenLearning.set_forecast_model(
    model,
    forecast_units(model_type == :uni ? nn : nns),
)

# Baseline metrics from the pretrained predictor.
ls_pred_train = deflatten_model_forecast(model.forecast(X_train')', model.forecast_vars)
ls_G_err_train = ls_pred_train[:, :, 1, :] - deflatten(Y_train_clean)[:, :, 1, :]
ls_G_rmse_train = mean(ls_G_err_train .^ 2) .^ 0.5
ls_Pi_err_train = ls_pred_train[:, :, 2, :] - deflatten(Y_train_clean)[:, :, 2, :]
ls_Pi_rmse_train = mean(ls_Pi_err_train .^ 2) .^ 0.5
ls_cost_train = ADL.compute_cost(model, X_train, Y_dict_train)

ls_pred_test = deflatten_model_forecast(model.forecast(X_test')', model.forecast_vars)
ls_G_err_test = ls_pred_test[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
ls_G_rmse_test = mean(ls_G_err_test .^ 2) .^ 0.5
ls_Pi_err_test = ls_pred_test[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
ls_Pi_rmse_test = mean(ls_Pi_err_test .^ 2) .^ 0.5
ls_cost_test = ADL.compute_cost(model, X_test, Y_dict_test)

println("Pretrained predictor train RMSE: G = $ls_G_rmse_train, Pi = $ls_Pi_rmse_train")
println("Pretrained predictor train Cost: $ls_cost_train")
println("Pretrained predictor test RMSE: G = $ls_G_rmse_test, Pi = $ls_Pi_rmse_test")
println("Pretrained predictor test Cost: $ls_cost_test")