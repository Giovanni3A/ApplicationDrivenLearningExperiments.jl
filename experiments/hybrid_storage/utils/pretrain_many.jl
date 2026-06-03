# Archetype :many — two shared networks (g_net, pi_net) applied once per project.
# Builds on :type by shrinking each subnet to consume one project's features at a
# time; ADL's input_output_map drives the per-project re-application with shared
# weights. The activations drop the per-position std/μ baked into :type's heads
# because the same activation runs for every project; the network is left to
# learn the absolute physical scale from its inputs.

const G_INPUT_SIZE_MANY = length(g_feat_idx_per_proj[1])
const PI_INPUT_SIZE_MANY = length(pi_feat_idx_per_proj[1])
const PER_PROJ_OUTPUT_SIZE = H

for p = 1:N_PROJ
    @assert length(g_feat_idx_per_proj[p]) == G_INPUT_SIZE_MANY
    @assert length(pi_feat_idx_per_proj[p]) == PI_INPUT_SIZE_MANY
end

const g_target_idx_per_proj = [[y_idx_G(p, h) for h = 1:H] for p = 1:N_PROJ]
const pi_target_idx_per_proj = [[y_idx_pi(p, h) for h = 1:H] for p = 1:N_PROJ]

function _build_subnet_many(input_size, output_size, output_activation)
    if N_HIDDEN_LAYERS == 0
        return Flux.Chain(
            Flux.Dense(input_size => output_size),
            output_activation,
        )
    end
    layers = Any[Flux.Dense(input_size => HIDDEN_SIZE), Flux.relu]
    for _ = 1:(N_HIDDEN_LAYERS-1)
        push!(layers, Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE))
        push!(layers, Flux.relu)
    end
    push!(layers, Flux.Dense(HIDDEN_SIZE => output_size))
    push!(layers, output_activation)
    return Flux.Chain(layers...)
end

nns = [
    _build_subnet_many(
        G_INPUT_SIZE_MANY,
        PER_PROJ_OUTPUT_SIZE,
        g_output_activation_many,
    ),
    _build_subnet_many(
        PI_INPUT_SIZE_MANY,
        PER_PROJ_OUTPUT_SIZE,
        pi_output_activation_many,
    ),
]

if pretrain
    batch_sz =
        PRETRAIN_BATCH_SIZE == -1 ? size(X_train, 1) : PRETRAIN_BATCH_SIZE
    train_data = Flux.DataLoader(
        (X_train', Y_train_clean'),
        batchsize = batch_sz,
        shuffle = true,
    )
    opt_state = Flux.setup(Flux.Adam(PRETRAIN_LEARNING_RATE), nns)
    local epoch = 1
    local init_time = time()
    local err = 1e18
    local stable_iters = 0
    while epoch <= PRETRAIN_EPOCHS
        Flux.train!(nns, train_data, opt_state) do m, x, y
            return sum(
                mean(
                    (
                        m[1](x[g_feat_idx_per_proj[p], :]) .-
                        y[g_target_idx_per_proj[p], :]
                    ) .^ 2,
                ) for p = 1:N_PROJ
            ) + sum(
                mean(
                    (
                        m[2](x[pi_feat_idx_per_proj[p], :]) .-
                        y[pi_target_idx_per_proj[p], :]
                    ) .^ 2,
                ) for p = 1:N_PROJ
            )
        end

        g_err = sum(
            mean(
                (
                    nns[1](X_train'[g_feat_idx_per_proj[p], :]) .-
                    Y_train_clean'[g_target_idx_per_proj[p], :]
                ) .^ 2,
            ) for p = 1:N_PROJ
        )
        pi_err = sum(
            mean(
                (
                    nns[2](X_train'[pi_feat_idx_per_proj[p], :]) .-
                    Y_train_clean'[pi_target_idx_per_proj[p], :]
                ) .^ 2,
            ) for p = 1:N_PROJ
        )
        err2 = g_err + pi_err
        err_var = abs(err - err2)
        err = err2

        if err_var < PRETRAIN_CONV_TOL
            stable_iters += 1
        else
            stable_iters = 0
        end
        if stable_iters > PRETRAIN_CONV_PATIENCE
            println("Pre-training converged after $epoch epochs.")
            break
        end

        if epoch % 10 == 0
            println(
                "Epoch $epoch | G MSE = $(round(g_err, digits=4)) | π MSE = $(round(pi_err, digits=4))",
            )
        end
        if (time() - init_time) > PRETRAIN_MAX_TIME
            println("Pre-training time limit reached at epoch $epoch.")
            break
        end

        epoch += 1
    end

    JLD2.jldsave(pretrained_model_state; state = Flux.state(nns))
    println("Pretrained state saved to $pretrained_model_state")
else
    models_state = JLD2.load(pretrained_model_state, "state")
    Flux.loadmodel!(nns, models_state)
    println("Loaded pretrained state from $pretrained_model_state")
end
