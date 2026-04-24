// RUN: nn-opt --pass-pipeline='builtin.module(lower-nn-to-standard)' %s | FileCheck %s

module {
  func.func @lowering(%input: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %weights = arith.constant dense<1.0> : tensor<3x4xf32>
    %bias = arith.constant dense<0.0> : tensor<4xf32>
    %dense = nn.dense %input, %weights, %bias : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32> -> tensor<2x4xf32>
    %relu = nn.relu %dense : tensor<2x4xf32> -> tensor<2x4xf32>
    return %relu : tensor<2x4xf32>
  }
}

// CHECK-NOT: nn.dense
// CHECK-NOT: nn.relu
// CHECK: linalg
// CHECK: arith
