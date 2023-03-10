set(LLVM_TARGET_DEFINITIONS Conversion.td)
mlir_tablegen(Conversion.hpp.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(NablaConversionIncGen)
