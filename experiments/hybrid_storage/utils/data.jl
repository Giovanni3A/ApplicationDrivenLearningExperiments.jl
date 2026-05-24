# Load joined_hourly_anon.csv → X/Y matrices for the hybrid_storage experiment.
#
# Conventions match matpower/utils/data.jl: top-level script (no explicit module),
# creates globals consumed by main.jl and the pretrain_*.jl files.
#
# Targets are built by forward-shifting <proj>_hsl and <proj>_pi by 1..H hours.
# Features are everything else except diagnostic and timestamp columns.
# Both X and Y are standardized with train-set statistics.

using CSV
using DataFrames
using Dates
using Statistics
using Random

Random.seed!(0)

const DATA_CSV = joinpath(@__DIR__, "..", "data", "joined_hourly_anon.csv")

# 1. Read CSV. Keep the timestamp column as String to handle the tz offset suffix.
df = CSV.read(DATA_CSV, DataFrame; types = Dict("interval_end_local" => String))

# 2. Parse timestamps to naive local DateTime. The CSV column carries a tz offset
#    (CDT/CST) which we strip after parsing — DST fall-back produces two rows with
#    the same naive hour, which is harmless for our window-based splits.
const TS_FMT = DateFormat("yyyy-mm-dd HH:MM:SS")
df.timestamp =
    DateTime.(
        replace.(df.interval_end_local, r"\s*[+-]\d\d:\d\d\s*$" => ""),
        TS_FMT,
    )
sort!(df, :timestamp)

# Select columns of interest
[i for i in Base.product(String.(PROJECT_IDS), [1, 2, 6, 24])]
feature_names = [
    # timestamp features
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    # system load forecast at start / mid / end of planning horizon
    "load_total_forecast1",
    "load_total_forecast12",
    "load_total_forecast24",
    # ERCOT-wide wind generation forecast at start / mid / end of horizon
    "ercot_powerwind_forecast1",
    "ercot_powerwind_forecast12",
    "ercot_powerwind_forecast24",
    # ERCOT-wide solar generation forecast at start / mid / end of horizon
    "ercot_powersolar_forecast1",
    "ercot_powersolar_forecast12",
    "ercot_powersolar_forecast24",
    # generation lags
    [
        "$(p)_hsl_lag$(n)" for
        (p, n) in Base.product(String.(PROJECT_IDS), [1, 2, 6, 24])
    ]...,
    # price lags
    [
        "$(p)_pi_lag$(n)" for
        (p, n) in Base.product(String.(PROJECT_IDS), [1, 2, 6, 24])
    ]...,
    # DAM prices
    ["$(p)_dam" for p in String.(PROJECT_IDS)]...,
]

# 5. Build target columns: <proj>_g_lead_h and <proj>_pi_lead_h for h = 1..H.
#    Layout matches the ADL forecast_vars order built in model.jl: all G first
#    (project outer, horizon inner), then all π in the same order. Lead-h target
#    at row i is the raw value at row i+h; last h rows get `missing`.
n = nrow(df)
for pid in PROJECT_IDS
    pids = String(pid)
    src_g = "$(pids)_hsl"
    src_pi = "$(pids)_pi"
    for h = 1:H
        df[!, "$(pids)_g_lead$h"] =
            vcat(df[(h+1):end, src_g], fill(missing, h))
        df[!, "$(pids)_pi_lead$h"] =
            vcat(df[(h+1):end, src_pi], fill(missing, h))
    end
end
target_names = String[]
for pid in PROJECT_IDS
    pids = String(pid)
    for h = 1:H
        push!(target_names, "$(pids)_g_lead$h")
    end
end
for pid in PROJECT_IDS
    pids = String(pid)
    for h = 1:H
        push!(target_names, "$(pids)_pi_lead$h")
    end
end

# 7. Replace missing with NaN and cast to Float32.
function tonumeric(col)::Vector{Float32}
    return Float32.(coalesce.(col, NaN))
end

# prices have to be clipped
for col in target_names
    if occursin("pi_lead", col)
        df[!, col] = clamp.(tonumeric(df[!, col]), -250, 5000)
    end
end


function build_xy(df_sub::SubDataFrame)
    X = reduce(hcat, [tonumeric(df_sub[!, c]) for c in feature_names])
    Y = reduce(hcat, [tonumeric(df_sub[!, c]) for c in target_names])
    return X, Y
end

# 8. Chronological split.
train_view =
    view(df, (df.timestamp .>= TRAIN_START) .& (df.timestamp .< TRAIN_END), :)
test_view =
    view(df, (df.timestamp .>= TEST_START) .& (df.timestamp .< TEST_END), :)

X_train_raw, Y_train_raw = build_xy(train_view)
X_test_raw, Y_test_raw = build_xy(test_view)

# 9. Drop rows where any target is NaN (the trailing H hours of each split, where
#    lead targets can't be materialized). Keep rows with NaN features — those get
#    median-imputed in step 10.
function target_complete(Y)
    return vec(.!any(isnan, Y, dims = 2))
end
train_keep = target_complete(Y_train_raw)
test_keep = target_complete(Y_test_raw)

X_train_dirty = X_train_raw[train_keep, :]
Y_train_clean = Y_train_raw[train_keep, :]
X_test_dirty = X_test_raw[test_keep, :]
Y_test_clean = Y_test_raw[test_keep, :]

# 10. Median-impute features. Median is computed on the train slice only; the same
#     vector imputes both train and test (no leakage). NaN columns (none expected
#     after the all-missing drop above, but defensively) get filled with 0.
function nanmedian(v)
    finite = v[.!isnan.(v)]
    return isempty(finite) ? 0.0f0 : Float32(median(finite))
end
feature_medians =
    Float32[nanmedian(X_train_dirty[:, j]) for j in axes(X_train_dirty, 2)]

function impute!(X, medians)
    for j in axes(X, 2)
        col = view(X, :, j)
        @inbounds for i in eachindex(col)
            if isnan(col[i])
                col[i] = medians[j]
            end
        end
    end
end
X_train_clean = copy(X_train_dirty);
impute!(X_train_clean, feature_medians);
X_test_clean = copy(X_test_dirty);
impute!(X_test_clean, feature_medians);

# 11. Standardize using train statistics. Zero-variance columns get std=1 so the
#     transformation is the identity (matches matpower's data.jl).
mu_X = mean(X_train_clean, dims = 1)
std_X = std(X_train_clean, dims = 1)
std_X[std_X.==0.0f0] .= 1.0f0

mu_Y = mean(Y_train_clean, dims = 1)
std_Y = std(Y_train_clean, dims = 1)
std_Y[std_Y.==0.0f0] .= 1.0f0

X_train = (X_train_clean .- mu_X) ./ std_X
X_test = (X_test_clean .- mu_X) ./ std_X
Y_train = (Y_train_clean .- mu_Y) ./ std_Y
Y_test = (Y_test_clean .- mu_Y) ./ std_Y

println("data.jl loaded:")
println("  features: $(length(feature_names))")
println("  targets:  $(length(target_names))")
println("  X_train: $(size(X_train))    Y_train: $(size(Y_train))")
println("  X_test:  $(size(X_test))     Y_test:  $(size(Y_test))")

# Helpers: indices into Y for each (project, horizon, variable). Y is laid out as
# all G entries (project outer, horizon inner), then all π entries the same way —
# matching the ADL forecast_vars order assembled in model.jl.
y_idx_G(pidx, h) = (pidx - 1) * H + h
y_idx_pi(pidx, h) = N_PROJ * H + (pidx - 1) * H + h

# Reshape a flat target row (or matrix of rows) into a per-(hour, channel, project)
# view. Y columns are laid out as all G (p outer, h inner) then all π same way;
# reshape naturally gives (..., H, N_PROJ, 2), and we permute so callers can keep
# using the (..., H, 2, N_PROJ) — i.e. channel-before-project — indexing.
function deflatten(x::AbstractMatrix)
    return PermutedDimsArray(
        reshape(x, size(x, 1), H, N_PROJ, 2),
        (1, 2, 4, 3),
    )
end

function deflatten(x::AbstractVector)
    return PermutedDimsArray(reshape(x, H, N_PROJ, 2), (1, 3, 2))
end

# Output reparametrization: enforce G_phys >= 0 inside the network instead of
# via LP slack penalties. Maps z_norm → ((softplus(z_norm·std + μ) − μ)/std) on G
# channels and passes π channels through. Output stays in normalized space so the
# LP's un-standardization and the MSE pretraining loss are unchanged.
const _G_OUT_MASK = let m = falses(length(target_names))
    for pidx = 1:N_PROJ, h = 1:H
        m[y_idx_G(pidx, h)] = true
    end
    Float32.(m)
end
const _MU_Y_VEC = Float32.(vec(mu_Y))
const _STD_Y_VEC = Float32.(vec(std_Y))

# --- Per-target activations for the multi (G/π split) model ---
# Output indices ordered to match vec(G_fc) / vec(pi_fc) (column-major: h varies, then p).
const _G_TARGET_IDX = vec([y_idx_G(p, h) for h = 1:H, p = 1:N_PROJ])
const _PI_TARGET_IDX = vec([y_idx_pi(p, h) for h = 1:H, p = 1:N_PROJ])

const _STD_Y_G = _STD_Y_VEC[_G_TARGET_IDX]
const _STD_Y_PI = _STD_Y_VEC[_PI_TARGET_IDX]
const _MU_Y_PI = _MU_Y_VEC[_PI_TARGET_IDX]

# Feature partitioning (calendar shared; gen lags + ERCOT-wide renewable forecasts → G;
# price lags + DAM + system load forecast → π).
const _CAL_FEAT_NAMES =
    ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend"]
const _GEN_LAG_FEAT_NAMES = vec([
    "$(p)_hsl_lag$(n)" for
    (p, n) in Base.product(String.(PROJECT_IDS), [1, 2, 6, 24])
])
const _PI_LAG_FEAT_NAMES = vec([
    "$(p)_pi_lag$(n)" for
    (p, n) in Base.product(String.(PROJECT_IDS), [1, 2, 6, 24])
])
const _DAM_FEAT_NAMES = ["$(p)_dam" for p in String.(PROJECT_IDS)]
const _LOAD_FORECAST_FEAT_NAMES =
    ["load_total_forecast$(h)" for h in [1, 12, 24]]
const _WIND_FORECAST_FEAT_NAMES =
    ["ercot_powerwind_forecast$(h)" for h in [1, 12, 24]]
const _SOLAR_FORECAST_FEAT_NAMES =
    ["ercot_powersolar_forecast$(h)" for h in [1, 12, 24]]

const g_feat_names = vcat(
    _CAL_FEAT_NAMES,
    _GEN_LAG_FEAT_NAMES,
    _WIND_FORECAST_FEAT_NAMES,
    _SOLAR_FORECAST_FEAT_NAMES,
)
const pi_feat_names = vcat(
    _CAL_FEAT_NAMES,
    _PI_LAG_FEAT_NAMES,
    _DAM_FEAT_NAMES,
    _LOAD_FORECAST_FEAT_NAMES,
)
const g_feat_idx = findall(in(g_feat_names), feature_names)
const pi_feat_idx = findall(in(pi_feat_names), feature_names)

# Per-project feature subsets for the :many (shared-net, applied-per-project) model.
# ERCOT-wide forecasts are system-level and shared across all projects (same columns
# go into every per-project slice), keeping the input dimension constant across projects.
const g_feat_names_per_proj = [
    vcat(
        _CAL_FEAT_NAMES,
        ["$(String(p))_hsl_lag$(n)" for n in [1, 2, 6, 24]],
        _WIND_FORECAST_FEAT_NAMES,
        _SOLAR_FORECAST_FEAT_NAMES,
    ) for p in PROJECT_IDS
]
const pi_feat_names_per_proj = [
    vcat(
        _CAL_FEAT_NAMES,
        ["$(String(p))_pi_lag$(n)" for n in [1, 2, 6, 24]],
        ["$(String(p))_dam"],
        _LOAD_FORECAST_FEAT_NAMES,
    ) for p in PROJECT_IDS
]
const g_feat_idx_per_proj =
    [findall(in(names), feature_names) for names in g_feat_names_per_proj]
const pi_feat_idx_per_proj =
    [findall(in(names), feature_names) for names in pi_feat_names_per_proj]

# G activation: softplus * std (matches the G branch of nonneg_g_output — no μ added).
function g_output_activation(z::AbstractArray)
    return Flux.softplus.(z) .* _STD_Y_G
end

# π activation: regular affine (no positivity constraint).
function pi_output_activation(z::AbstractArray)
    return z .* _STD_Y_PI .+ _MU_Y_PI
end

# Shared activations for :many — same network is applied across projects, so std/μ
# must be project-agnostic. We average across projects per hour, giving a 24-vector
# that approximates the right physical scale for every project. The network learns
# the residual per-project deviation from this mean.
const _STD_Y_G_PERHOUR = Float32[
    mean(_STD_Y_VEC[y_idx_G(p, h)] for p = 1:N_PROJ) for h = 1:H
]
const _STD_Y_PI_PERHOUR = Float32[
    mean(_STD_Y_VEC[y_idx_pi(p, h)] for p = 1:N_PROJ) for h = 1:H
]
const _MU_Y_PI_PERHOUR = Float32[
    mean(_MU_Y_VEC[y_idx_pi(p, h)] for p = 1:N_PROJ) for h = 1:H
]

function g_output_activation_many(z::AbstractArray)
    return Flux.softplus.(z) .* _STD_Y_G_PERHOUR
end

function pi_output_activation_many(z::AbstractArray)
    return z .* _STD_Y_PI_PERHOUR .+ _MU_Y_PI_PERHOUR
end

function concat_z_activation(z::AbstractArray)
    z_phys = z .* _STD_Y_VEC .+ _MU_Y_VEC
    z_norm_pos = Flux.softplus.(z) .* _STD_Y_VEC
    return _G_OUT_MASK .* z_norm_pos .+ (1.0f0 .- _G_OUT_MASK) .* z_phys
end
