#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  using namespace mlir;
  
  DialectRegistry registry;
  registerAllDialects(registry);

  return failed(MlirOptMain(argc, argv, "Nabla modular optimizer driver\n", registry));
}