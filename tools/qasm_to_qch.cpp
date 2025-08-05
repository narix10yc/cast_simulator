#include "cast/Legacy/CircuitGraph.h"
#include "cast/Parser.h"
#include "openqasm/Parser.h"
#include "utils/CommandLine.h"

#include <filesystem>
namespace fs = std::filesystem;

static const auto& ArgInputFile =
    utils::cl::registerArgument<std::string>("input-name")
        .desc("input file/directory name")
        .setArgumentPositional();

static const auto& ArgOutputFile =
    utils::cl::registerArgument<std::string>("o").desc("output file name");

static const auto& ArgIsDirectory =
    utils::cl::registerArgument<bool>("r").desc("recursive").init(false);

enum ConversionResult {
  ResultSuccess,
  ErrCannotOpenInput,
  ErrCannotOpenOutput
};

[[nodiscard]] ConversionResult convert(const std::string& inName,
                                       const std::string& outName) {
  std::cerr << "converting " << inName << " to " << outName << std::endl;
  // read file
  std::ifstream inFile(inName);
  if (!inFile.is_open()) {
    std::cerr << "Could not open input file '" << inName << "'.\n";
    return ErrCannotOpenInput;
  }
  openqasm::Parser qasmParser(inName, 0);
  auto qasmRoot = qasmParser.parse();
  cast::legacy::CircuitGraph graph;
  qasmRoot->toCircuitGraph(graph);
  auto qc = cast::ast::QuantumCircuit::FromCircuitGraph(graph);
  inFile.close();

  // write file
  std::ofstream outFile(outName);
  if (!outFile.is_open()) {
    std::cerr << "Could not open output file '" << outName << "'.\n";
    return ErrCannotOpenOutput;
  }
  qc.print(outFile);
  outFile.close();
  return ResultSuccess;
}

int main(int argc, char** argv) {
  utils::cl::ParseCommandLineArguments(argc, argv);
  utils::cl::DisplayArguments();

  if (!fs::exists(static_cast<std::string>(ArgInputFile))) {
    std::cerr << "Input file (directory) '" << ArgInputFile
              << "' does not exist. Exiting...\n";
    return 1;
  }

  if (fs::is_directory(static_cast<std::string>(ArgInputFile))) {
    if (!ArgIsDirectory) {
      std::cerr << "'" << ArgInputFile
                << "' seems to be a directory? "
                   "Use -r for recursive conversion.\n";
      return 1;
    }
    int nFilesProcessed = 0;
    int nSuccess = 0;
    for (const auto& f :
         fs::directory_iterator(static_cast<std::string>(ArgInputFile))) {

      const auto fName = f.path().filename().string();
      if (!f.is_regular_file()) {
        std::cerr << "Omitted " << fName
                  << " because it is not a regular file\n";
        continue;
      }
      const auto fNameLength = fName.length();
      if (fNameLength <= 5 || fName.substr(fNameLength - 5, 5) != ".qasm") {
        std::cerr << "Omitted " << fName
                  << " because its name does not end with '.qasm'\n";
        continue;
      }

      auto ofName = fs::path(static_cast<std::string>(ArgOutputFile)) /
                    (fName.substr(0, fNameLength - 5) + ".qch");

      nFilesProcessed++;
      auto rst = convert(f.path().string(), ofName);
      if (rst == ResultSuccess)
        nSuccess++;
    }
    if (nSuccess == nFilesProcessed) {
      std::cerr << nSuccess << " files processed.\n";
      return 0;
    } else {
      std::cerr << nSuccess << " out of " << nFilesProcessed
                << " files successfully processed!\n";
      return 1;
    }
  } else {
    auto rst = convert(ArgInputFile, ArgOutputFile);
    if (rst != ResultSuccess) {
      std::cerr << "Failed. Exiting...\n";
      return 1;
    }
  }
  utils::cl::unregisterAllArguments();
  return 0;
}