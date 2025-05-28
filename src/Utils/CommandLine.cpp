#include "utils/CommandLine.h"
#include <algorithm>
#include <vector>

using namespace utils;
using namespace utils::cl::internal;

ArgumentRegistry ArgumentRegistry::Instance;

static int levenshteinDistance(StringRef s1, StringRef s2) {
  size_t len1 = s1.length(), len2 = s2.length();
  std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

  for (size_t i = 0; i <= len1; ++i)
    dp[i][0] = i;
  for (size_t j = 0; j <= len2; ++j)
    dp[0][j] = j;

  for (size_t i = 1; i <= len1; ++i) {
    for (size_t j = 1; j <= len2; ++j) {
      if (s1[i - 1] == s2[j - 1])
        dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
    }
  }
  return dp[len1][len2];
}

[[noreturn]]
static void terminateDisplayHelp() {
  cl::DisplayHelp();
  std::exit(0);
}

static void warnTooManyHyphens(int nHyphens, StringRef name) {
  std::cerr << BOLDYELLOW("[Warning] ") << "Argument '" << name << "' "
            "is prefixed by " << nHyphens << " (> 2) hyphens.\n";
}

static void warnEqualSignInPrefixArg(char prefix) {
  std::cerr << BOLDYELLOW("[Warning] ") << "Prefix argument '" << prefix << "' "
            "should not be followed by '='.\n";
}

[[noreturn]]
static void terminateFailToParseArgument(StringRef clName, StringRef clValue) {
  std::cerr << BOLDRED("[Error] ") << "Argument '" << clName
            << "' does not accept value '" << clValue << "'\n";
  std::exit(1);
}

[[noreturn]]
static void terminateLeadingEqualSign(StringRef clValue) {
  std::cerr << BOLDRED("[Error] ") << "Leading equal sign in '" << clValue
            << "'; Did you add a redundant space before it?\n";
  std::exit(1);
}

[[noreturn]]
static void terminateNoProvidingRequiredValue(StringRef name) {
  std::cerr << BOLDRED("[Error] ") << "Argument '" << name << "' "
            "requires a value, but is not provided.\n";
  std::exit(1);
}

[[noreturn]]
static void terminateNoPositionalArgument(StringRef clInput) {
  std::cerr << BOLDRED("[Error] ")
            << "There is no positional argument defined for input '" << clInput
            << "'\n";
  std::exit(1);
}

[[noreturn]]
static void terminateEmptyArgument() {
  std::cerr << BOLDRED("[Error] ") << "Empty argument\n";
  std::exit(1);
}

[[noreturn]]
static void terminateUnknownArgument(StringRef arg) {
  std::cerr << BOLDRED("[Error] ") << "Unknown argument '" << arg << "'. ";
  int minDist = 100;
  StringRef bestMatchArg;
  for (const auto* a: ArgumentRegistry::arguments()) {
    StringRef candidateArg(a->getName());
    auto candidateDist = levenshteinDistance(candidateArg, arg);
    if (candidateDist < minDist) {
      minDist = candidateDist;
      bestMatchArg = candidateArg;
    }
  }

  if (minDist < 6)
    std::cerr << "Did you mean '" << bestMatchArg << "'?";
  std::cerr << "\n";
  std::exit(1);
}

[[noreturn]]
static void terminateSpaceAfterEqualSign(
    StringRef clName, StringRef clValue) {
  std::cerr << BOLDRED("[Error] ") << "Argument-value pair '" << clName << "= "
            << clValue << "' should not have space after the '=' sign.\n";
  std::exit(1);
}

namespace {
  struct ArgContainer {
    ArgumentBase* arg;
    int nOccurrence;

    explicit ArgContainer(ArgumentBase* a) : arg(a), nOccurrence(0) {}
  };
}

static void maybeTerminateNonMatchOccurrence(const ArgContainer& argContainer) {
  if (argContainer.arg == nullptr)
    return;
  assert(argContainer.nOccurrence >= 0);
  switch (argContainer.arg->getOccurrenceFormat()) {
  case cl::OccurExactlyOnce: {
    if (argContainer.nOccurrence != 1) {
      std::cerr << BOLDRED("[Error] ")
                << "Argument '" << argContainer.arg->getName()
                << "' needs to be specified exactly once, but it was ";
      if (argContainer.nOccurrence == 0)
        std::cerr << "not specified.\n";
      else
        std::cerr << "specified " << argContainer.nOccurrence << " times.\n";
      std::exit(1);
    }
    break;
  }
  case cl::OccurAtLeastOnce: {
    if (argContainer.nOccurrence == 0) {
      std::cerr << BOLDRED("[Error] ")
                << "Argument '" << argContainer.arg->getName()
                << "' should be specified at least once.\n";
      std::exit(1);
    }
    break;
  }
  case cl::OccurAtMostOnce: {
    if (argContainer.nOccurrence > 1) {
      std::cerr << BOLDRED("[Error] ")
                << "Argument '" << argContainer.arg->getName()
                << "' should be specified at most once, but was specified "
                << argContainer.nOccurrence << " times.\n";
      std::exit(1);
    }
    break;
  }
  default:
    break;
  }
}

void cl::ParseCommandLineArguments(int argc, char** argv) {
  std::vector<ArgContainer> prefixArgs;
  std::vector<ArgContainer> nonPrefixArgs;

  ArgContainer positionalArgContainer(nullptr);
  for (auto* a : ArgumentRegistry::arguments()) {
    switch (a->getArgFormat()) {
      case AF_Positional: {
        assert(positionalArgContainer.arg == nullptr &&
          "At most one positional arg is allowed");
        positionalArgContainer.arg = a;
        break;
      }
      case AF_Prefix: {
        prefixArgs.emplace_back(a);
        assert(a->getName().length() == 1 &&
          "Prefix arguments must have name with length 1");
        #ifdef DISALLOW_LOWERCASE_PREFIX_ARGUMENT_NAMES
          assert(*a->getName().begin() >= 'A' && *a->getName().begin() <= 'Z' &&
           "Prefix argument names must be a single upper-case letter. "
           "To disable this, undef DISALLOW_LOWERCASE_PREFIX_ARGUMENT_NAMES "
           "immediately after including Commandline.h and re-compile. "
           "(Unexpected parsing results may happen if this lower-case letter "
           "collides with other argument names.)");
        #endif
        break;
      }
      default:
        nonPrefixArgs.emplace_back(a);
    }
  }

  const auto matchPrefixArg = [&](char prefix) -> ArgContainer* {
    for (auto& c : prefixArgs) {
    if (*c.arg->getName().begin() == prefix)
        return &c;
    }
    return nullptr;
  };

  const auto matchNonPrefixArg = [&](StringRef name) -> ArgContainer* {
    for (auto& c : nonPrefixArgs) {
      if (c.arg->getName().compare(name) == 0)
        return &c;
    }
    return nullptr;
  };

  const auto parseAndRecordOccurrence =
    [&](ArgContainer& arg, StringRef clValue) {
      if (arg.arg->parseValue(clValue))
        terminateFailToParseArgument(arg.arg->getName(), clValue);
      arg.nOccurrence++;
  };

  int i = 1;
  /* main loop
   * In every iteration, we will match a name-value pair.
   */
  while (i < argc) {
    StringRef clInput(argv[i++]);
    assert(!clInput.empty());

    if (*clInput.begin() == '=') {
      terminateLeadingEqualSign(clInput);
      // terminated
    }

    // skip hyphens to grab the true argument name
    int nHyphens = 0;
    while (*clInput.begin() == '-') {
      clInput.increment();
      nHyphens++;
    }
    if (clInput.empty()) {
      terminateEmptyArgument();
      // terminated
    }

    if (nHyphens > 2)
      warnTooManyHyphens(nHyphens, clInput);

    if (clInput.compare("help") == 0) {
      terminateDisplayHelp();
      // terminated
    }

    // positional arguments, the whole clInput is clValue
    if (nHyphens == 0) {
      if (positionalArgContainer.arg == nullptr) {
        terminateNoPositionalArgument(clInput);
        // terminated
      }
      parseAndRecordOccurrence(positionalArgContainer, clInput);
      continue;
    }

    // split name-value pair by '='. value is empty iff there is no '=' sign
    StringRef clName;
    StringRef clValue(clInput);
    while (!clValue.empty() && *clValue.begin() != '=')
      clValue.increment();
    clName = StringRef(clInput.begin(), clValue.begin() - clInput.begin());
    assert(!clName.empty() &&
      "Name should not be empty. "
      "It should be checked by 'terminateLeadingEqualSign'");
    if (!clValue.empty()) {
      assert(*clValue.begin() == '=');
      clValue.increment(); // go past the '=' sign
      if (clValue.empty()) {
        terminateSpaceAfterEqualSign(clName, (i < argc) ? argv[i+1] : "");
        // terminated
      }
    }

    /* If the user defines an arg 'output-dir' and a prefix arg 'o', then 
     * '-output-dir=my/dir' should be parsed as (output-dir, my/dir) instead of
     * (o, utput-dir=my/dir).
     * So the above name-value split will handle this case correctly.
     * On the other hand, if 'output-dir' is not defined, we should use 
     * (o, utput-dir=my/dir) instead. We handle it in the following.
     */
    ArgContainer* argContainer = nullptr;
    if ((argContainer = matchNonPrefixArg(clName)) != nullptr) {
      // Successfully matched a non-prefix arg. Parse and finish.
      auto vf = argContainer->arg->getValueFormat();
      if (vf == VF_Required) {
        if (clValue.empty()) {
          if (i == argc)
            terminateNoProvidingRequiredValue(clName);
          clValue = argv[i++];
          if (*clValue.begin() == '-')
            terminateNoProvidingRequiredValue(clName);
        }
        parseAndRecordOccurrence(*argContainer, clValue);
        continue;
      }
      assert(vf == VF_Optional);
      if (clValue.empty() && i != argc) {
        clValue = argv[i];
        // invalid value
        if (*clValue.begin() == '-')
          clValue = "";
      }
      parseAndRecordOccurrence(*argContainer, clValue);
      if (!clValue.empty())
        ++i;
      continue;
    }

    // prefix arguments
    argContainer = matchPrefixArg(*clInput.begin());
    if (argContainer == nullptr) {
      terminateUnknownArgument(clName);
      // terminated
    }
    // re-split the name-value pair
    clName = StringRef(clInput.begin(), 1);
    clValue = StringRef(clInput.begin() + 1, clInput.length() - 1);
    if (!clValue.empty() && *clValue.begin() == '=') {
      warnEqualSignInPrefixArg(*clName.begin());
      clValue.increment();
    }
    if (clValue.empty()) {
      if (i == argc)
        terminateNoProvidingRequiredValue(clName);
      clValue = argv[i++];
    }
    parseAndRecordOccurrence(*argContainer, clValue);
  } // main while loop for parsing

  // occurrence check
  maybeTerminateNonMatchOccurrence(positionalArgContainer);
  for (const auto& argContainer : prefixArgs)
    maybeTerminateNonMatchOccurrence(argContainer);
  for (const auto& argContainer : nonPrefixArgs)
    maybeTerminateNonMatchOccurrence(argContainer);
}

void cl::DisplayArguments() {
  std::cerr << "----------------- " << ArgumentRegistry::arguments().size()
            << " Arguments -----------------\n";
  for (const auto* a: ArgumentRegistry::arguments()) {
    auto argName = a->getName();
    std::cerr << argName << ": ";
    if (argName.length() < 30)
      std::cerr << std::string(30 - argName.length(), ' ');
    a->printValue(std::cerr);
    std::cerr << "\n";
  }
  std::cerr << "--------------- End of Arguments ---------------\n";
}

void cl::DisplayHelp() {
  std::cerr << "<-help>\n";
}

void cl::unregisterAllArguments() {
  for (const auto* a: ArgumentRegistry::arguments())
    delete a;
  ArgumentRegistry::arguments().clear();
}

template<>
bool cl::ArgumentParser<std::string>::operator()(
    StringRef clValue, std::string& valueToWriteOn) {
      valueToWriteOn = static_cast<std::string>(clValue);
  return false;
}

template<>
bool cl::ArgumentParser<int>::operator()(
    StringRef clValue, int& valueToWriteOn) {
  valueToWriteOn = std::stoi(static_cast<std::string>(clValue));
  return false;
}

template<>
bool cl::ArgumentParser<double>::operator()(
    StringRef clValue, double& valueToWriteOn) {
  valueToWriteOn = std::stod(static_cast<std::string>(clValue));
  return false;
}

template<>
bool cl::ArgumentParser<bool>::operator()(
    StringRef clValue, bool& valueToWriteOn) {
  if (clValue.compare("0") == 0 || clValue.compare_ci("false") == 0 ||
      clValue.compare_ci("off") == 0) {
    valueToWriteOn = false;
    return false;
  }
  if (clValue.empty() || clValue.compare("1") == 0 || 
      clValue.compare_ci("true") == 0 || clValue.compare_ci("on") == 0) {
    valueToWriteOn = true;
    return false;
  }
  return true;
}

