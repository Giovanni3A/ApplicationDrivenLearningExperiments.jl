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
#    Layout per project: [G_lead1..G_lead24, π_lead1..π_lead24] (block-per-quantity,
#    so the per-project output slice is the first 24 cols = G and next 24 = π).
#    Lead-h target at row i is the raw value at row i+h. Last h rows get `missing`.
n = nrow(df)
target_names = String[]
for pid in PROJECT_IDS
    pids = String(pid)
    src_g = "$(pids)_hsl"
    src_pi = "$(pids)_pi"
    for h = 1:H
        g_col = "$(pids)_g_lead$h"
        df[!, g_col] = vcat(df[(h+1):end, src_g], fill(missing, h))
        push!(target_names, g_col)
    end
    for h = 1:H
        pi_col = "$(pids)_pi_lead$h"
        df[!, pi_col] = vcat(df[(h+1):end, src_pi], fill(missing, h))
        push!(target_names, pi_col)
    end
end

# 7. Replace missing with NaN and cast to Float32.
function tonumeric(col)::Vector{Float32}
    return Float32.(coalesce.(col, NaN))
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

# Helpers: indices into Y for each (project, horizon, variable).
y_idx_G(pidx, h) = (pidx - 1) * N_OUTPUTS_PER_PROJ + h
y_idx_pi(pidx, h) = (pidx - 1) * N_OUTPUTS_PER_PROJ + H + h

# Un-standardized physical values (variables × scalar constants → linear).
function _G_phys_plan(h, pidx)
    return G_fc[h, pidx].plan * std_Y[y_idx_G(pidx, h)] + mu_Y[y_idx_G(pidx, h)]
end
function _G_phys_assess(h, pidx)
    return G_fc[h, pidx].assess * std_Y[y_idx_G(pidx, h)] +
           mu_Y[y_idx_G(pidx, h)]
end
function _pi_phys_plan(h, pidx)
    return pi_fc[h, pidx].plan * std_Y[y_idx_pi(pidx, h)] +
           mu_Y[y_idx_pi(pidx, h)]
end
function _pi_phys_assess(h, pidx)
    return pi_fc[h, pidx].assess * std_Y[y_idx_pi(pidx, h)] +
           mu_Y[y_idx_pi(pidx, h)]
end

function deflatten(x::AbstractMatrix)
    return reshape(x, size(x, 1), H, 2, N_PROJ)
end

function deflatten(x::AbstractVector)
    return reshape(x, H, 2, N_PROJ)
end

function denormalize_and_deflatten(x::AbstractMatrix)
    return deflatten(x .* std_Y .+ mu_Y)
end

function denormalize_and_deflatten(x::AbstractVector)
    return deflatten(x .* std_Y .+ mu_Y)
end

deflattened_Y_train = deflatten(Y_train_clean)
deflattened_Y_test = deflatten(Y_test_clean)