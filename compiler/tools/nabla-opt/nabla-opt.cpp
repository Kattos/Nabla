#include "Conversion/Conversion.hpp"
#include "Dialect/Nabla/IR/Nabla.hpp"
#include "Dialect/Nabla/IR/NablaInterface.hpp"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  using namespace mlir;

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<nabla::NablaDialect>();

  nabla::registerConversionPasses();
  nabla::registerAdjointInterface(registry);

  return failed(
      MlirOptMain(argc, argv, "Nabla modular optimizer driver\n", registry));
}