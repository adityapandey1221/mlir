// RUN: nn-opt %s | FileCheck %s

module {
  func.func @roundtrip(%input: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %weights = arith.constant dense<1.0> : tensor<3x4xf32>
    %bias = arith.constant dense<0.0> : tensor<4xf32>
    %dense = nn.dense %input, %weights, %bias : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
    %relu = nn.relu %dense : tensor<2x4xf32> -> tensor<2x4xf32>
    return %relu : tensor<2x4xf32>
  }
}

// CHECK: module
// CHECK: func.func @roundtrip
// CHECK: nn.dense
// CHECK: nn.relu
