if model_type == :uni
    include("pretrain_uni.jl")
elseif model_type == :type
    include("pretrain_type.jl")
elseif model_type == :many
    include("pretrain_many.jl")
else
    error("Invalid model_type (:uni, :type, :many)")
end
