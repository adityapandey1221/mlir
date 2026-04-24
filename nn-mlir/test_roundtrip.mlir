module {
  func.func @simple(%input: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %weights = arith.constant dense<1.0> : tensor<3x4xf32>
    %bias = arith.constant dense<0.0> : tensor<4xf32>
    %output = nn.dense %input, %weights, %bias : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
    return %output : tensor<2x4xf32>
  }
}
