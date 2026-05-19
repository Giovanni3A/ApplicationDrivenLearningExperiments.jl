# Experiment configuration for hybrid_storage.
# Follows matpower's config.jl pattern: toggles + hyperparams + paths.

# Predictive model archetype
model_type = 1   # 1 = uni (one shared model), 2 = many (per-project models), 3 = multi (joint multivariate)
pretrain = false

# Network architecture
N_HIDDEN_LAYERS = 2      # 0 = linear; >0 = MLP with ReLU
HIDDEN_SIZE = 64

# Pretrain hyperparameters
PRETRAIN_EPOCHS = 1000
PRETRAIN_MAX_TIME = 60 * 10         # seconds
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_BATCH_SIZE = 256            # -1 = full batch
PRETRAIN_CONV_TOL = 1e-7
PRETRAIN_CONV_PATIENCE = 30

# Decision-focused training hyperparameters (used when run_mode == 2)
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
COMPUTE_EVERY = 10
TIME_LIMIT = 60 * 60

# Output paths
const ARCH_NAME = model_type == 1 ? "uni" : model_type == 2 ? "many" : "multi"
result_path = joinpath(@__DIR__, "results", "size_$(N_HIDDEN_LAYERS)")
isdir(result_path) || mkpath(result_path)

pretrained_model_state = joinpath(result_path, "pretrain_$(ARCH_NAME).jld2")
final_model_state = joinpath(result_path, "gradient_$(ARCH_NAME).jld2")
