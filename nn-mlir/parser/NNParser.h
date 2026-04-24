#ifndef NN_PARSER_H
#define NN_PARSER_H

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace nn_frontend {

// Data structure to store a singular dense layer
struct DenseLayer {
  unsigned units = 0;
  bool applyRelu = false;
};

// Data structure to store the neural network as a vector of dense layers
struct Network {
  std::string name;
  unsigned inputSize = 0;
  std::vector<DenseLayer> layers;
};

/* ----------HELPER FUNCTIONS----------- */

// Trims leading and trailing whitespaces from the input string
inline std::string trim(std::string text) {
  auto notSpace = [](unsigned char c) { return !std::isspace(c); };
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), notSpace));
  text.erase(std::find_if(text.rbegin(), text.rend(), notSpace).base(),
             text.end());
  return text;
}

// Strips code comments
inline std::string stripComment(const std::string &line) {
  const std::size_t hashPos = line.find('#');
  const std::size_t slashPos = line.find("//");

  std::size_t cut = std::string::npos;
  if (hashPos != std::string::npos) {
    cut = hashPos;
  }
  if (slashPos != std::string::npos) {
    cut = cut == std::string::npos ? slashPos : std::min(cut, slashPos);
  }

  return cut == std::string::npos ? line : line.substr(0, cut);
}

// Parses a positive integer token. Example: "input 784", "dense 256"
inline bool parseUnsignedToken(const std::string &token, unsigned &value) {
  if (token.empty() ||
      !std::all_of(token.begin(), token.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      })) {
    return false;
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(token.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<unsigned>::max()) {
    return false;
  }

  value = static_cast<unsigned>(parsed);
  return true;
}

/* ---------- CORE PARSER ----------- */

// Parses the neural network
// Uses a state machine for parsing instead of recursive descent parser since the DSL is line-based and flat, and 
// has no scoped elements like functions, loops, if statements, and other code blocks
inline bool parseNetworkText(const std::string &text, Network &network,
                             std::string &error) {
  std::istringstream stream(text);
  std::string rawLine;
  std::size_t lineNumber = 0;

  enum class State { Start, SawNetwork, SawInput };
  State state = State::Start;
  unsigned currentSize = 0;
  
  // Iterate through file lines, apply helper functions to clean the text before parsing
  while (std::getline(stream, rawLine)) {

    // Iterate through file lines, apply helper functions to clean the text before parsing
    ++lineNumber;
    std::string line = trim(stripComment(rawLine));
    if (line.empty()) {
      continue;
    }

    std::istringstream lineStream(line);
    std::string keyword;
    lineStream >> keyword;
    
    // If parsed token is "network", start the parsing of the network, change parser state to SawNetwork
    if (keyword == "network") {
      if (state != State::Start) {
        error = "line " + std::to_string(lineNumber) +
                ": `network` must appear once at the top of the file";
        return false;
      }

      if (!(lineStream >> network.name)) {
        error = "line " + std::to_string(lineNumber) +
                ": expected `network <name>`";
        return false;
      }

      std::string extra;
      if (lineStream >> extra) {
        error = "line " + std::to_string(lineNumber) +
                ": unexpected token after network name";
        return false;
      }

      state = State::SawNetwork;
      continue;
    }

    // If parsed token is "input", parse the input size, update parser state to SawInput
    if (keyword == "input") {
      if (state != State::SawNetwork) {
        error = "line " + std::to_string(lineNumber) +
                ": `input` must follow the `network` line";
        return false;
      }

      std::string sizeToken;
      if (!(lineStream >> sizeToken) || !parseUnsignedToken(sizeToken, currentSize)) {
        error = "line " + std::to_string(lineNumber) +
                ": expected `input <positive-size>`";
        return false;
      }

      std::string extra;
      if (lineStream >> extra) {
        error = "line " + std::to_string(lineNumber) +
                ": unexpected token after input size";
        return false;
      }

      network.inputSize = currentSize;
      state = State::SawInput;
      continue;
    }

    // If parsed token is "dense", parse the layer output size, create a DenseLayer, push to network.layers
    // Optionally accepts relu. relu is an addition onto the dense layer, not a separate token type itself
    if (keyword == "dense") {
      if (state != State::SawInput) {
        error = "line " + std::to_string(lineNumber) +
                ": `dense` must follow an `input` declaration";
        return false;
      }

      std::string unitsToken;
      if (!(lineStream >> unitsToken) ||
          !parseUnsignedToken(unitsToken, currentSize)) {
        error = "line " + std::to_string(lineNumber) +
                ": expected `dense <positive-units> [relu]`";
        return false;
      }

      DenseLayer layer;
      layer.units = currentSize;

      std::string maybeRelu;
      if (lineStream >> maybeRelu) {
        if (maybeRelu != "relu") {
          error = "line " + std::to_string(lineNumber) +
                  ": only optional `relu` is supported after dense units";
          return false;
        }
        layer.applyRelu = true;

        std::string extra;
        if (lineStream >> extra) {
          error = "line " + std::to_string(lineNumber) +
                  ": unexpected token after `relu`";
          return false;
        }
      }

      network.layers.push_back(layer);
      continue;
    }

    error = "line " + std::to_string(lineNumber) +
            ": unknown statement `" + keyword + "`";
    return false;
  }

  if (state == State::Start) {
    error = "missing `network <name>` declaration";
    return false;
  }

  if (state != State::SawInput) {
    error = "missing `input <size>` declaration";
    return false;
  }

  if (network.layers.empty()) {
    error = "network must contain at least one `dense` layer";
    return false;
  }

  return true;
}

} // namespace nn_frontend

#endif // NN_PARSER_H
