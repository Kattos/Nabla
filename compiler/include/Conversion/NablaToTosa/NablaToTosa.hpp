#ifndef NABLA_CONVERSION_TO_TOSA_HPP
#define NABLA_CONVERSION_TO_TOSA_HPP

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nabla {

#define GEN_PASS_DEF_NABLATOTOSA
#define GEN_PASS_DECL_NABLATOTOSA
#include "Conversion/Conversion.hpp.inc"

std::unique_ptr<Pass> createNablaToTosa();

}  // namespace nabla
}  // namespace mlir

#endif  // NABLA_CONVERSION_TO_TOSA_HPP
