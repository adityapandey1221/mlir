//===- NNOps.td - NN dialect operation definitions -*-tablegen-*-===//
// This file defines the operations for the NN dialect.
//===------------------------------------------------------------===//

#ifndef NN_OPS
#define NN_OPS

include "mlir/IR/OpBase.td" //imports MLIR's base definitions (Dialect, Op, etc.)

//===----------------------------------------------------------------------===//
// NN Dialect Definition
//===----------------------------------------------------------------------===//

def NN_Dialect : Dialect {
    let name = "nn";
    let summary = "A neural network dialect for MLIR";
    let cppNamespace = "::mlir::nn";
}

//===----------------------------------------------------------------------===//
// Base NN Operation
//===----------------------------------------------------------------------===//

// Template/base class 
// Params: 
//  mnemonic: name of operation
//  traits: optional list of operation traits
class NN_Op<string mnemonic, list<Trait> traits = []> : Op<NN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// NN Operations
//===----------------------------------------------------------------------===//

def DenseOp : NN_Op<"dense"> { //mnemonic is "dense"; becomes nn.dense in IR
    let summary = "Dense (fully-connected) layer: output = input @ weights + bias";
    let description = [{
        Performs a dense layer computation:
        output = input @ weights + bias

        Arguments:
        - input:   tensor<batch x in_features x f32>
        - weights: tensor<in_features x out_features x f32>
        - bias:    tensor<out_features x f32>

        Result:
        - output:  tensor<batch x out_features x f32>
    }];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$weights,
        AnyRankedTensor:$bias
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = [{
        $input `,` $weights `,` $bias attr-dict `:`
        type($input) `,` type($weights) `,` type($bias) `->` type($output)
    }];
}

def ReluOp : NN_Op<"relu"> {
    let summary = "ReLU activation: output = max(0, input)";
    let description = [{
        Applies element-wise ReLU activation:
        output = max(0, input)

        Input and output have the same tensor type.
    }];

    let arguments = (ins
        AnyRankedTensor:$input
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif // NN_OPS