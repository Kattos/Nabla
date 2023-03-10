add_subdirectory(Adjoint)
add_subdirectory(NablaToArith)
add_subdirectory(NablaToTosa)

add_llvm_library(NablaConversion
    Conversion.cpp

    DEPENDS
    AdjointConversion
    NablaToArith
    NablaToTosa
)