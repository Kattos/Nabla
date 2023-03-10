add_mlir_conversion_library(AdjointConversion
    Adjoint.cpp

    DEPENDS
    NablaConversionIncGen
)
