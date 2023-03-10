#include "Conversion/Adjoint/Adjoint.hpp"
#include "Conversion/NablaToArith/NablaToArith.hpp"
#include "Conversion/NablaToTosa/NablaToTosa.hpp"

#define GEN_PASS_REGISTRATION
#include "Conversion/Conversion.hpp.inc"

namespace mlir {
namespace nabla {
void registerConversionPasses() { ::registerConversionPasses(); }
}  // namespace nabla
}  // namespace mlir
