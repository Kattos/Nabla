add_subdirectory(AdjointInterface)

add_mlir_dialect_library(NablaDialect
    Nabla.cpp
    NablaInterface.cpp

    DEPENDS
    MLIRNablaIncGen
)
