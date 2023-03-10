#ifndef NABLA_CONVERSION_PASSES
#define NABLA_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def NablaToTosa : Pass<"convert-nabla-to-tosa"> {
    let summary = "Convert Nabla dialect to Tosa dialect";

    let constructor = "mlir::nabla::createNablaToTosa()";
    let dependentDialects = ["mlir::tosa::TosaDialect"];
}

def NablaToArith : Pass<"convert-nabla-to-arith"> {
    let summary = "Convert Nabla dialect to Arith dialect";

    let constructor = "mlir::nabla::createNablaToArith()";
    let dependentDialects = ["mlir::arith::ArithDialect"];
}

def AdjointPass : Pass<"adjoint"> {
    let summary = "Lower `nabla.adjoint` to dialects in mlir library";

    let constructor = "mlir::nabla::createAdjointPass()";
    let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // NABLA_CONVERSION_PASSES
