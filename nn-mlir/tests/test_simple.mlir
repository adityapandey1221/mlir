// Simple test: matrix multiply followed by relu
func.func @simple_network(%input: tensor<2x3xf32>, %weight: tensor<3x4xf32>, %bias: tensor<4xf32>) -> tensor<2x4xf32> {
  // Dense operation: matrix multiply + bias
  %dense_result = nn.dense %input, %weight, %bias : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>

  // ReLU operation: activation
  %relu_result = nn.relu %dense_result : tensor<2x4xf32> -> tensor<2x4xf32>

  return %relu_result : tensor<2x4xf32>
}
