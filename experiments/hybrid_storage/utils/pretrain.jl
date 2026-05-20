if model_type == :uni
    include("pretrain_uni.jl")
elseif model_type == :type
    include("pretrain_type.jl")
else
    error("Invalid model_type (:uni, :type)")
end
