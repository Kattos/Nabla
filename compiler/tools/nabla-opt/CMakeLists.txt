get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    NablaConversion
    NablaAdjointInterface
)

add_llvm_executable(nabla-opt nabla-opt.cpp)
target_link_libraries(nabla-opt PRIVATE ${LIBS})
