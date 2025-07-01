module SHAInet
  abstract class UnifiedMatrix
    abstract def forward(input : UnifiedMatrix) : UnifiedMatrix
    abstract def backward(grad : UnifiedMatrix) : UnifiedMatrix
    abstract def update_weights(lr : Float64)
  end
end

