#ifndef NABLA_INTERFACE
#define NABLA_INTERFACE

include "mlir/IR/OpBase.td"

// ηζδΌ΄ι op
def Nabla_AdjointInterface : OpInterface<"AdjointInterface"> {
    let cppNamespace = "::mlir::nabla";

    let methods = [
        InterfaceMethod<
            /*desc*/ "generate adjoint operation",
            /*retTy*/ "::mlir::SmallVector<::mlir::Value>",
            /*methodName*/ "adjoint",
            /*args*/ (ins "::mlir::Value": $dtarget, "::mlir::OpBuilder&": $builder)
        >,
        InterfaceMethod<
            "generate default derivative of target",
            "::mlir::Value",
            "defaultDerivative",
            (ins "::mlir::OpBuilder&": $builder)
        >
    ];
}

#endif // NABLA_INTERFACE
