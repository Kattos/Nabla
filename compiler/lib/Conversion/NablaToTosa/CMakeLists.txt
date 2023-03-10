add_mlir_conversion_library(NablaToTosa
    NablaToTosa.cpp

    DEPENDS
    NablaConversionIncGen
)