#ifndef NABLA_OPS
#define NABLA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Nabla_Dialect : Dialect {
    let name = "nabla";
    let cppNamespace = "::mlir::nabla";

    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Nabla_Op<string mnemonic, list<Trait> traits = []> : Op<Nabla_Dialect, mnemonic, traits>;

// 为正向 F(X) = Y 生成伴随 F*(X, Y*) = X*
def Nabla_AdjointOp : Nabla_Op<"adjoint"> {
    let summary = "generate adjoint of given operation";

    let arguments = (ins
        AnyType: $target,
        Optional<AnyType>: $dtarget
    );
    
    let results = (outs
        Variadic<AnyType>: $dsources
    );

    let hasVerifier = 1;
}

def Nabla_AccumulateOp : Nabla_Op<"accumulate", [SameOperandsAndResultType]> {
    let summary = "accumulate operation";

    let arguments = (ins
        Variadic<AnyType>: $inputs
    );

    let results = (outs
        AnyType: $accumulation
    );

    let hasVerifier = 1;
}

// 从给定值开始进行反向传播，类似于 torch.Tensor.backward
def Nabla_BackpropOp : Nabla_Op<"backprop", [SameOperandsShape, SameOperandsElementType]> {
    let summary = "backprop from target";

    let arguments = (ins
        AnyType: $target,
        Optional<AnyType>: $dtarget
    );
}

// 计算所有后向 op 的梯度贡献
def Nabla_GradientOp : Nabla_Op<"gradient", [SameOperandsAndResultType]> {
    let summary = "sum gradient contributions from value users";

    let arguments = (ins
        AnyType: $target
    );

    let results = (outs
        AnyType: $dtarget
    );
}

#endif // NABLA_OPS
