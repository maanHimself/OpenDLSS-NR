// dlss5vk: standalone Vulkan DLSS-NR runner.
//   dlss5vk parity --model <nr model dir> --fixture <fixtures/nr512> [--repeat N] [--shaders <dir>] [--dump <dir>]
//   dlss5vk bench  --model <nr model dir> --width W --height H [--frames N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <array>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "vk_context.h"

namespace {

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint8_t> bytes((size_t)size);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

std::string readText(const std::string& path) {
  std::vector<uint8_t> bytes = readFile(path);
  return std::string(bytes.begin(), bytes.end());
}

std::string argValue(int argc, char** argv, const char* name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}

bool hasFlag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return true;
  return false;
}

std::string executableDirectory(const char* argv0) {
  std::string path = argv0;
  size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

// After a frame: a chained wait that gave up means its output is wrong (docs/execution.md, "Barrier-free chaining").
void checkChainTimeouts(const nr::Kernels& kernels) {
  const nr::Kernels::ChainTimeouts timeouts = kernels.chainTimeouts();
  if (timeouts.waits)
    throw std::runtime_error(std::to_string(timeouts.waits) + " chained wait(s) timed out, the first on " + timeouts.counter +
                             ": the frame is invalid (DLSS5VK_CHAIN=0 runs without chaining)");
}

// DLSS5VK_UNFUSED=1 runs the least fused form of the graph (the GLSL reference kernels).
static bool fusedBlocksEnabled() {
  const char* unfused = getenv("DLSS5VK_UNFUSED");
  return !(unfused && !strcmp(unfused, "1"));
}

// ---- parity ----------------------------------------------------------------------------------------------------
// A fixture declares the comparisons it exists for ("checks") and carries one reference per comparison. The whole
// declaration is validated before the GPU runs and nothing is ever dropped from a count: a declared check that
// cannot run, a missing, malformed or unknown reference, or a reference no declared check uses rejects the fixture.
// Every verdict names the equality it proved (docs/numerics.md): bit-exact, equal only up to the sign of zero (a
// failure: not the same bytes), or within a stated tolerance.

struct BoundaryReference { std::string name, file; uint32_t width = 0, height = 0, channels = 0; };

struct FixturePlan {
  uint32_t validWidth = 0, validHeight = 0, fullWidth = 0, fullHeight = 0;
  std::string proxyFile, featuresFile;
  uint32_t proxyWidth = 0, proxyHeight = 0;
  bool checkBoundaries = false, checkHead = false, checkOutput = false;
  std::vector<BoundaryReference> boundaries;                  // graph order
  std::vector<std::pair<std::string, std::string>> omitted;   // boundary, why the fixture has no reference for it
  std::string headFile, outputFile;
  bool outputBytes = false;                                   // an RGBA8 image (a tolerance check), not RGBA f32 halves
};

int64_t fileSize(const std::string& path) {
  std::error_code error;
  const uintmax_t size = std::filesystem::file_size(path, error);
  return error ? -1 : (int64_t)size;
}

FixturePlan planFixture(const json::Value& manifest, const std::string& fixtureDir) {
  FixturePlan plan;
  std::vector<std::string> problems;
  auto fail = [&](const std::string& text) { problems.push_back(text); };
  auto dims = [&](const char* key, uint32_t& width, uint32_t& height) {
    if (!manifest.has(key) || manifest[key].kind != json::Value::Array || manifest[key].size() != 2) {
      fail(std::string("\"") + key + "\" must be [width, height]");
      return;
    }
    width = (uint32_t)manifest[key][0].integer(); height = (uint32_t)manifest[key][1].integer();
  };
  auto positive = [&](const std::string& what, const json::Value& entry, const char* key) -> uint32_t {
    if (!entry.has(key) || entry[key].integer() <= 0) { fail(what + ": \"" + key + "\" missing or not positive"); return 0; }
    return (uint32_t)entry[key].integer();
  };
  // A reference file must exist and hold exactly the bytes its shape says.
  auto reference = [&](const std::string& what, const json::Value& entry, int64_t bytes) -> std::string {
    if (!entry.has("file") || entry["file"].str().empty()) { fail(what + ": no \"file\""); return ""; }
    const std::string path = fixtureDir + "/" + entry["file"].str();
    const int64_t size = fileSize(path);
    if (size < 0) fail(what + ": cannot read " + path);
    else if (size != bytes) fail(what + ": " + path + " holds " + std::to_string(size) + " bytes, its shape needs " + std::to_string(bytes));
    return path;
  };
  dims("sourceDimensions", plan.validWidth, plan.validHeight);
  dims("fullDimensions", plan.fullWidth, plan.fullHeight);
  const int64_t fullRows = (int64_t)plan.fullWidth * plan.fullHeight;

  // The input: the proxy image the features are generated from, or the recorded features themselves.
  if (manifest.has("proxy") == manifest.has("inputFeatures")) fail("exactly one of \"proxy\" and \"inputFeatures\" is the input");
  if (manifest.has("proxy")) {
    const json::Value& proxy = manifest["proxy"];
    plan.proxyWidth = positive("proxy", proxy, "width");
    plan.proxyHeight = positive("proxy", proxy, "height");
    plan.proxyFile = reference("proxy", proxy, (int64_t)plan.proxyWidth * plan.proxyHeight * 16);
    for (const char* key : {"conditioning", "seed", "autoMask"})
      if (!manifest.has(key)) fail(std::string("a proxy fixture needs \"") + key + "\"");
  } else if (manifest.has("inputFeatures")) {
    plan.featuresFile = reference("inputFeatures", manifest["inputFeatures"], fullRows * 16 * 4);
  }

  // The declared check set.
  std::set<std::string> checks;
  if (!manifest.has("checks") || manifest["checks"].kind != json::Value::Array || manifest["checks"].size() == 0) {
    fail("\"checks\" must list what the fixture gates: \"boundaries\", \"head\", \"output\"");
  } else {
    for (const json::Value& check : manifest["checks"].array) {
      if (check.str() != "boundaries" && check.str() != "head" && check.str() != "output") fail("unknown check \"" + check.str() + "\"");
      else if (!checks.insert(check.str()).second) fail("check \"" + check.str() + "\" listed twice");
    }
  }
  plan.checkBoundaries = checks.count("boundaries") != 0;
  plan.checkHead = checks.count("head") != 0;
  plan.checkOutput = checks.count("output") != 0;

  // Boundaries: every stored output the graph has a reference name for is either compared or declared omitted,
  // with a reason, so a shortened export cannot pass as a smaller suite.
  const bool hasBoundaryKeys = (manifest.has("blocks") && manifest["blocks"].size()) ||
                               (manifest.has("transitions") && manifest["transitions"].size()) || manifest.has("omittedBoundaries");
  if (plan.checkBoundaries) {
    std::map<std::string, BoundaryReference> byName;
    auto add = [&](const std::string& name, const json::Value& entry) {
      BoundaryReference ref{name};
      ref.width = positive(name, entry, "width"); ref.height = positive(name, entry, "height");
      ref.channels = positive(name, entry, "channels");
      ref.file = reference(name, entry, (int64_t)ref.width * ref.height * ref.channels);
      if (!byName.emplace(name, ref).second) fail(name + " listed twice");
    };
    if (manifest.has("blocks"))
      for (const json::Value& entry : manifest["blocks"].array) {
        if (!entry.has("block")) { fail("a \"blocks\" entry has no \"block\""); continue; }
        add("block-" + std::to_string(entry["block"].integer()), entry);
      }
    if (manifest.has("transitions"))
      for (const json::Value& entry : manifest["transitions"].array) {
        if (!entry.has("id")) { fail("a \"transitions\" entry has no \"id\""); continue; }
        add("transition-" + entry["id"].str(), entry);
      }
    std::map<std::string, std::string> omitted;
    if (manifest.has("omittedBoundaries"))
      for (const auto& [name, reason] : manifest["omittedBoundaries"].object) {
        if (reason.str().empty()) fail("omitted boundary " + name + " gives no reason");
        omitted[name] = reason.str();
      }
    const std::vector<std::string>& names = nr::Graph::referenceBoundaryNames();
    for (const auto& [name, ref] : byName)
      if (std::find(names.begin(), names.end(), name) == names.end()) fail(name + " is not a boundary of this graph");
    for (const auto& [name, reason] : omitted) {
      if (std::find(names.begin(), names.end(), name) == names.end()) fail("omitted " + name + " is not a boundary of this graph");
      if (byName.count(name)) fail(name + " is both compared and declared omitted");
    }
    std::vector<std::string> unaccounted;
    for (const std::string& name : names) {
      if (byName.count(name)) plan.boundaries.push_back(byName[name]);
      else if (omitted.count(name)) plan.omitted.emplace_back(name, omitted[name]);
      else unaccounted.push_back(name);
    }
    if (!unaccounted.empty()) {
      std::string list;
      for (size_t i = 0; i < unaccounted.size() && i < 8; ++i) list += (i ? ", " : "") + unaccounted[i];
      fail(std::to_string(unaccounted.size()) + " of " + std::to_string(names.size()) + " graph boundaries have neither a reference nor an omission reason (" +
           list + (unaccounted.size() > 8 ? ", ..." : "") + ")");
    }
  } else if (hasBoundaryKeys) {
    fail("the fixture carries boundary references but does not declare the \"boundaries\" check");
  }

  if (plan.checkHead != manifest.has("referenceHead"))
    fail(plan.checkHead ? "the \"head\" check needs \"referenceHead\"" : "\"referenceHead\" is carried but the \"head\" check is not declared");
  else if (plan.checkHead)
    plan.headFile = reference("referenceHead", manifest["referenceHead"], fullRows * 16);

  if (plan.checkOutput != manifest.has("nativeOutput")) {
    fail(plan.checkOutput ? "the \"output\" check needs \"nativeOutput\"" : "\"nativeOutput\" is carried but the \"output\" check is not declared");
  } else if (plan.checkOutput) {
    const json::Value& output = manifest["nativeOutput"];
    const std::string dtype = output.has("dtype") ? output["dtype"].str() : "";
    const uint32_t width = positive("nativeOutput", output, "width"), height = positive("nativeOutput", output, "height");
    if (width != plan.validWidth || height != plan.validHeight) fail("nativeOutput is not the source size");
    if (dtype == "f32") {
      if (plan.proxyFile.empty()) fail("an f32 nativeOutput is composed from the proxy, and the fixture has none");
      plan.outputFile = reference("nativeOutput", output, (int64_t)width * height * 16);
    } else if (dtype == "u8") {
      plan.outputBytes = true;
      plan.outputFile = reference("nativeOutput", output, (int64_t)width * height * 4);
    } else {
      fail("nativeOutput \"dtype\" must be \"f32\" (RGBA f32 halves) or \"u8\" (RGBA8)");
    }
  }

  if (!problems.empty()) {
    for (const std::string& problem : problems) fprintf(stderr, "fixture: %s\n", problem.c_str());
    throw std::runtime_error("fixture rejected (" + std::to_string(problems.size()) + " problem" + (problems.size() == 1 ? "" : "s") +
                             "); nothing was run");
  }
  return plan;
}

// One comparison, element by element. `signedZeros` counts elements that differ only in the sign of a zero;
// `tolerated` those within the check's stated tolerance; `differing` everything else.
struct Tally {
  size_t count = 0, differing = 0, signedZeros = 0, tolerated = 0;
  size_t first = SIZE_MAX;
  double maxAbs = 0, sumSquares = 0;
  std::map<int, size_t> codeDeltas;   // E4M3: signed code distance, clamped to +-9
  void miss(size_t index, double difference) {
    if (first == SIZE_MAX) first = index;
    ++differing;
    maxAbs = std::max(maxAbs, std::fabs(difference));
    sumSquares += difference * difference;
  }
};

Tally compareE4(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected) {
  Tally t;
  t.count = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    const uint8_t a = actual[i], e = expected[i];
    if (a == e) continue;
    if ((a & 0x7f) == 0 && (e & 0x7f) == 0) { ++t.signedZeros; continue; }
    t.miss(i, (double)num::e4m3ToF32(a) - num::e4m3ToF32(e));
    const int signedA = (a & 0x80) ? -(int)(a & 0x7f) : (int)(a & 0x7f);
    const int signedE = (e & 0x80) ? -(int)(e & 0x7f) : (int)(e & 0x7f);
    t.codeDeltas[std::max(-9, std::min(9, signedA - signedE))]++;
  }
  return t;
}

// f32 elements by bit pattern: +0 and -0 differ, a NaN equals only the same NaN.
Tally compareBits(const float* actual, const float* expected, size_t count) {
  Tally t;
  t.count = count;
  for (size_t i = 0; i < count; ++i) {
    uint32_t a, e;
    memcpy(&a, actual + i, 4); memcpy(&e, expected + i, 4);
    if (a == e) continue;
    if (((a | e) & 0x7fffffffu) == 0) { ++t.signedZeros; continue; }
    t.miss(i, (double)actual[i] - expected[i]);
  }
  return t;
}

// Truncation (toward zero) to the half grid: the composite's publication of the neural result.
float truncateHalf(float value) {
  uint32_t bits; memcpy(&bits, &value, 4);
  const uint32_t signBit = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 0xffu, mantissa = bits & 0x7fffffu;
  uint32_t halfBits;
  if (exponent == 0xffu) halfBits = signBit | (mantissa ? 0x7e00u : 0x7c00u);
  else {
    const int halfExponent = (int)exponent - 112;
    if (halfExponent >= 31) halfBits = signBit | 0x7c00u;
    else if (halfExponent <= 0) halfBits = halfExponent < -10 ? signBit : signBit | ((mantissa | 0x800000u) >> (14 - halfExponent));
    else halfBits = signBit | ((uint32_t)halfExponent << 10) | (mantissa >> 13);
  }
  return num::f16ToF32((uint16_t)halfBits);
}

// The composed RGB of one frame with no history, as the demo composite publishes it:
// neural = clamp((head / 32 + centred) * 8 + 0.5, 0, 1), truncated to the half grid. `inner * 8` is exact, so there
// is one rounding whether or not the last multiply-add is contracted.
float composed(float head, float centred) {
  const float inner = std::fmaf(head, 0.03125f, centred);
  return truncateHalf(std::fmin(std::fmax(inner * 8.0f + 0.5f, 0.0f), 1.0f));
}

Tally compareOutput(const FixturePlan& plan, const float* head, const std::vector<uint8_t>& input, const std::vector<uint8_t>& reference) {
  const float* values = reinterpret_cast<const float*>(input.data());
  Tally t;
  t.count = (size_t)plan.validWidth * plan.validHeight * 3;
  for (uint32_t y = 0; y < plan.validHeight; ++y)
    for (uint32_t x = 0; x < plan.validWidth; ++x)
      for (uint32_t c = 0; c < 3; ++c) {
        const float h = head[((size_t)y * plan.fullWidth + x) * 4 + c];
        const size_t pixel = (size_t)y * plan.validWidth + x;
        if (plan.outputBytes) {
          // An RGBA8 capture is the same image quantized to eight bits by a rounding this repository does not know
          // exactly, so the check is a tolerance: one code either way. The centred proxy is the features' lane 4 + c.
          const float centred = values[((size_t)y * plan.fullWidth + x) * 16 + 4 + c];
          const int published = (int)std::fmin(255.0f, std::fmax(0.0f, std::floor(composed(h, centred) * 255.0f + 0.5f)));
          const int expected = reference[pixel * 4 + c];
          if (published == expected) continue;
          if (std::abs(published - expected) == 1) { ++t.tolerated; continue; }
          t.miss(pixel * 3 + c, published - expected);
        } else {
          const float centred = std::fmaf(values[pixel * 4 + c], 0.125f, -0.0625f);
          const float published = composed(h, centred);
          const float expected = reinterpret_cast<const float*>(reference.data())[pixel * 4 + c];
          const Tally one = compareBits(&published, &expected, 1);
          t.signedZeros += one.signedZeros;
          if (one.differing) t.miss(pixel * 3 + c, (double)published - expected);
        }
      }
  return t;
}

enum class Verdict { BitExact, SignedZeroOnly, WithinTolerance, Mismatch };

Verdict verdictOf(const Tally& t) {
  if (t.differing) return Verdict::Mismatch;
  if (t.signedZeros) return Verdict::SignedZeroOnly;
  return t.tolerated ? Verdict::WithinTolerance : Verdict::BitExact;
}

std::string describe(const Tally& t) {
  char text[256];
  switch (verdictOf(t)) {
    case Verdict::BitExact: snprintf(text, sizeof(text), "bit-exact (%zu)", t.count); break;
    case Verdict::SignedZeroOnly:
      snprintf(text, sizeof(text), "NOT BIT-EXACT: %zu of %zu differ only in the sign of zero", t.signedZeros, t.count); break;
    case Verdict::WithinTolerance:
      snprintf(text, sizeof(text), "within one code (%zu exact, %zu one code off)", t.count - t.tolerated, t.tolerated); break;
    case Verdict::Mismatch:
      snprintf(text, sizeof(text), "MISMATCH %zu/%zu (%.4f%%) max|d| %.6g rmse %.6g%s", t.differing, t.count, 100.0 * t.differing / t.count,
               t.maxAbs, std::sqrt(t.sumSquares / t.count), t.signedZeros ? " (+ signed zeros)" : "");
      break;
  }
  return text;
}

// A graph built for one schedule, recorded and submitted `submissions` times on the same buffers (as the demo
// resubmits its frame); `repeatable` says whether every submission gave the first one's head.
struct GraphRun {
  std::vector<uint8_t> head;
  std::map<std::string, std::vector<uint8_t>> boundaries;
  double minGpuMs = 1e30;
  uint32_t dispatches = 0;
  bool chained = false, repeatable = true;
};

GraphRun runGraph(vk::Context& context, nr::Model& model, nr::Kernels& kernels, const nr::Geometry& geometry,
                  const nr::Activation& features, bool chain, bool capture, int submissions) {
  const bool chainBefore = nr::Kernels::chainEnabled();
  nr::Kernels::setChainEnabled(chainBefore && chain);
  GraphRun run;
  {
    nr::Graph::Options options;
    options.captureBoundaries = capture;
    options.fusedBlocks = fusedBlocksEnabled();
    nr::Graph graph(context, model, kernels, geometry, options);
    run.chained = graph.chained();
    VkQueryPool queries = context.createTimestampPool(2);
    for (int submission = 0; submission < submissions; ++submission) {
      context.resetDescriptorPool();
      VkCommandBuffer commands = context.beginCommands();
      vkCmdResetQueryPool(commands, queries, 0, 2);
      vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
      graph.record(commands, features);
      vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
      context.endAndSubmit(commands, true);
      checkChainTimeouts(kernels);
      std::vector<double> stamps = context.readTimestampsMs(queries, 2);
      run.minGpuMs = std::min(run.minGpuMs, stamps[1] - stamps[0]);
      run.dispatches = kernels.dispatchCount();
      std::vector<uint8_t> head = context.download(graph.head().buffer, graph.head().validBytes());
      if (submission == 0) run.head = std::move(head);
      else run.repeatable = run.repeatable && head == run.head;
    }
    vkDestroyQueryPool(context.device(), queries, nullptr);
    for (const auto& [name, activation] : graph.boundaries())
      run.boundaries[name] = context.download(activation->buffer, activation->validBytes());
  }
  nr::Kernels::setChainEnabled(chainBefore);
  return run;
}

int runParity(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string fixtureDir = argValue(argc, argv, "--fixture");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  std::string dumpDir = argValue(argc, argv, "--dump");
  const int repeats = std::max(2, atoi(argValue(argc, argv, "--repeat", "3").c_str()));
  if (modelDir.empty() || fixtureDir.empty()) {
    fprintf(stderr, "usage: dlss5vk parity --model <dir> --fixture <dir> [--repeat N] [--shaders <dir>] [--dump <dir>]\n");
    return 2;
  }
  json::Value manifest = json::parse(readText(fixtureDir + "/manifest.json"));
  const FixturePlan plan = planFixture(manifest, fixtureDir);

  vk::Context context;
  printf("device: %s\n", context.deviceName().c_str());
  auto started = std::chrono::steady_clock::now();
  nr::Model model(context, modelDir, !hasFlag(argc, argv, "--no-verify"));
  printf("model loaded in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(plan.validWidth, plan.validHeight);
  printf("geometry %ux%u -> full %ux%u, levels", plan.validWidth, plan.validHeight, geometry.fullWidth, geometry.fullHeight);
  for (auto level : geometry.levels) printf(" %ux%u", level.width, level.height);
  printf("\n");
  if (geometry.fullWidth != plan.fullWidth || geometry.fullHeight != plan.fullHeight)
    throw std::runtime_error("fixture full dimensions disagree with the runtime profile");
  printf("checks:%s%s%s\n", plan.checkBoundaries ? " boundaries" : "", plan.checkHead ? " head" : "", plan.checkOutput ? " output" : "");
  if (plan.checkBoundaries) {
    printf("  boundaries: %zu references, %zu declared omitted\n", plan.boundaries.size(), plan.omitted.size());
    std::map<std::string, std::vector<std::string>> byReason;
    for (const auto& [name, reason] : plan.omitted) byReason[reason].push_back(name);
    for (const auto& [reason, names] : byReason)
      printf("    omitted (%s): %s%s%s\n", reason.c_str(), names.front().c_str(), names.size() > 1 ? " .. " : "",
             names.size() > 1 ? names.back().c_str() : "");
  }
  if (plan.checkOutput) printf("  output: %s\n", plan.outputBytes ? "RGBA8, compared within one code" : "RGBA f32 halves, bit-exact");

  // The features the graph reads, generated from the proxy on the GPU or uploaded as recorded.
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation features;
  features.format = nr::Format::F32; features.rows = fullRows; features.channels = 16; features.allocRows = nr::alignRows(fullRows);
  features.label = "input features";
  features.buffer = context.createBuffer((VkDeviceSize)features.allocRows * 16 * 4, false, "input features");
  context.fillZero(features.buffer);
  std::vector<uint8_t> inputBytes = readFile(plan.proxyFile.empty() ? plan.featuresFile : plan.proxyFile);
  vk::Buffer proxyBuffer;
  if (!plan.proxyFile.empty()) {
    proxyBuffer = context.createBuffer(inputBytes.size(), false, "proxy");
    context.upload(proxyBuffer, inputBytes.data(), inputBytes.size());
    const json::Value& conditioning = manifest["conditioning"];
    nr::Kernels::PreprocessArgs preprocess{geometry.fullWidth, geometry.fullHeight, plan.validWidth, plan.validHeight,
                                           plan.proxyWidth, plan.proxyHeight, (uint32_t)manifest["seed"].integer(),
                                           manifest["autoMask"].boolean, (float)conditioning["localTone"].number,
                                           (float)conditioning["localStructure"].number, (float)conditioning["skinStructure"].number,
                                           (float)conditioning["style"].number};
    VkCommandBuffer commands = context.beginCommands();
    kernels.preprocessFromProxy(commands, proxyBuffer, features, preprocess);
    context.endAndSubmit(commands, true);
  } else {
    context.upload(features.buffer, inputBytes.data(), inputBytes.size());
  }
  if (!dumpDir.empty()) {
    std::vector<uint8_t> bytes = context.download(features.buffer, features.validBytes());
    std::ofstream(dumpDir + "/features.f32", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  size_t failures = 0, tolerated = 0, bitExact = 0;
  auto require = [&](bool ok, const char* what) {
    printf("%s: %s\n", what, ok ? "identical" : "DIFFERENT");
    if (!ok) ++failures;
  };

  // A first graph builds every matrix; then the raw tensor buffers keep only the bytes the kernels read through
  // them (Model::compactAux), and every check below runs on the compacted model.
  const size_t rawBefore = model.rawBytes();
  const GraphRun uncompacted = runGraph(context, model, kernels, geometry, features, true, false, 1);
  model.compactAux();
  printf("tensor raw buffers: %.1f MiB, %.1f KiB after compaction (matrices %.1f MiB)\n", rawBefore / 1048576.0,
         model.rawBytes() / 1024.0, model.matrixBytes() / 1048576.0);
  // The production schedule (no captures, chaining as configured) is what the head and output references are
  // compared against, resubmitted several times: its head must not change between submissions.
  const GraphRun production = runGraph(context, model, kernels, geometry, features, true, false, repeats);
  printf("production schedule (%s, no captures): %u dispatches, GPU %.3f ms (min of %d)\n",
         production.chained ? "counter chaining" : "barriers", production.dispatches, production.minGpuMs, repeats);
  printf("NaN weight codes replaced: %zu\n", model.nanWeightsReplaced());
  require(production.repeatable, ("head over " + std::to_string(repeats) + " production submissions").c_str());
  require(production.head == uncompacted.head, "head before and after compacting the raw tensor buffers");
  // Chaining is scheduling, not arithmetic: the same graph with a barrier after every launch gives the same head.
  if (production.chained)
    require(runGraph(context, model, kernels, geometry, features, false, false, 1).head == production.head, "head with barriers instead of chaining");
  else
    printf("chaining is off in this configuration: the barrier schedule is the production schedule\n");
  // Boundaries come from an instrumented schedule, which adds a copy and two barriers at every boundary (and
  // materializes the deferred projections); that must not change the head either.
  GraphRun instrumented;
  if (plan.checkBoundaries) {
    instrumented = runGraph(context, model, kernels, geometry, features, true, true, 1);
    require(instrumented.head == production.head, "head of the instrumented schedule (boundary captures) vs production");
  }

  printf("\n");
  auto count = [&](const Tally& t) {
    const Verdict v = verdictOf(t);
    if (v == Verdict::BitExact) ++bitExact;
    else if (v == Verdict::WithinTolerance) ++tolerated;
    else ++failures;
  };
  bool codeHistogramShown = false;
  for (const BoundaryReference& ref : plan.boundaries) {
    auto it = instrumented.boundaries.find(ref.name);
    if (it == instrumented.boundaries.end()) {
      printf("%-16s NOT CAPTURED by this route\n", ref.name.c_str());
      ++failures;
      continue;
    }
    const std::vector<uint8_t> expected = readFile(ref.file);
    if (it->second.size() != expected.size()) {
      printf("%-16s SHAPE: the graph has %zu bytes, the reference %zu\n", ref.name.c_str(), it->second.size(), expected.size());
      ++failures;
      continue;
    }
    if (!dumpDir.empty())
      std::ofstream(dumpDir + "/" + ref.name + ".u8", std::ios::binary).write(reinterpret_cast<const char*>(it->second.data()), it->second.size());
    const Tally t = compareE4(it->second, expected);
    count(t);
    printf("%-16s %ux%ux%u %s", ref.name.c_str(), ref.width, ref.height, ref.channels, describe(t).c_str());
    if (t.differing) {
      const size_t pixel = t.first / ref.channels;
      printf(" first@ x%zu y%zu c%zu: got 0x%02x exp 0x%02x", pixel % ref.width, pixel / ref.width, t.first % ref.channels,
             it->second[t.first], expected[t.first]);
      if (!codeHistogramShown) {
        printf("  code-delta histogram:");
        for (auto& [delta, n] : t.codeDeltas) printf(" %+d:%zu", delta, n);
        codeHistogramShown = true;
      }
    }
    printf("\n");
  }

  const float* head = reinterpret_cast<const float*>(production.head.data());
  const size_t headValues = (size_t)fullRows * 4;
  if (plan.checkHead) {
    const std::vector<uint8_t> reference = readFile(plan.headFile);
    const Tally t = compareBits(head, reinterpret_cast<const float*>(reference.data()), headValues);
    count(t);
    printf("head vs reference: %s", describe(t).c_str());
    if (t.differing) printf(" first@ x%zu y%zu c%zu", (t.first / 4) % geometry.fullWidth, (t.first / 4) / geometry.fullWidth, t.first % 4);
    printf("\n");
  }
  if (plan.checkOutput) {
    const Tally t = compareOutput(plan, head, inputBytes, readFile(plan.outputFile));
    count(t);
    printf("composed RGB vs native output: %s", describe(t).c_str());
    if (t.differing) printf(" first@ x%zu y%zu c%zu", (t.first / 3) % plan.validWidth, (t.first / 3) / plan.validWidth, t.first % 3);
    printf("\n");
  }
  double sum = 0, sumSquares = 0; size_t nonFinite = 0;
  for (size_t i = 0; i < headValues; ++i) {
    if (!std::isfinite(head[i])) { ++nonFinite; continue; }
    sum += head[i]; sumSquares += (double)head[i] * head[i];
  }
  printf("head: %zu values, mean %.6f, rms %.6f, non-finite %zu\n", headValues, sum / headValues, std::sqrt(sumSquares / headValues), nonFinite);
  if (!dumpDir.empty())
    std::ofstream(dumpDir + "/head.f32", std::ios::binary).write(reinterpret_cast<const char*>(production.head.data()), production.head.size());
  if (proxyBuffer.buffer) context.destroyBuffer(proxyBuffer);
  context.destroyBuffer(features.buffer);

  printf("\nVERDICT: %s - %zu bit-exact, %zu within tolerance, %zu failed\n", failures ? "FAIL" : "PASS", bitExact, tolerated, failures);
  return failures ? 1 : 0;
}

int runBench(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t width = (uint32_t)atoi(argValue(argc, argv, "--width", "768").c_str());
  uint32_t height = (uint32_t)atoi(argValue(argc, argv, "--height", "768").c_str());
  int frames = atoi(argValue(argc, argv, "--frames", "10").c_str());
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk bench --model <dir> [--width W --height H --frames N]\n"); return 2; }
  vk::Context context;
  printf("device: %s\n", context.deviceName().c_str());
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  std::vector<float> synthetic((size_t)fullRows * 16);
  for (size_t i = 0; i < synthetic.size(); ++i) synthetic[i] = num::roundF16(std::sin(i * 0.0017f) * 0.125f);
  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
  // Warm compile.
  {
    VkCommandBuffer commands = context.beginCommands();
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
    checkChainTimeouts(kernels);
  }
  const size_t rawBefore = model.rawBytes();
  model.compactAux();   // every matrix is built: the raw buffers keep only what the kernels read through them
  printf("tensor raw buffers: %.1f MiB, %.1f KiB after compaction\n", rawBefore / 1048576.0, model.rawBytes() / 1024.0);
  VkQueryPool queries = context.createTimestampPool(2);
  std::vector<double> samples;
  for (int frame = 0; frame < frames; ++frame) {
    context.resetDescriptorPool();
    VkCommandBuffer commands = context.beginCommands();
    vkCmdResetQueryPool(commands, queries, 0, 2);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    graph.record(commands, *features);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    context.endAndSubmit(commands, true);
    checkChainTimeouts(kernels);
    std::vector<double> stamps = context.readTimestampsMs(queries, 2);
    samples.push_back(stamps[1] - stamps[0]);
    printf("frame %d: %.3f ms GPU (%u dispatches)\n", frame, samples.back(), kernels.dispatchCount());
  }
  std::sort(samples.begin(), samples.end());
  printf("median %.3f ms, min %.3f ms over %d frames at %ux%u (full %ux%u)\n", samples[samples.size() / 2],
         samples.front(), frames, width, height, geometry.fullWidth, geometry.fullHeight);
  vkDestroyQueryPool(context.device(), queries, nullptr);
  return 0;
}

}  // namespace

int runVerify(int argc, char** argv);

namespace {
int runShaderInfo(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  std::string filter = argValue(argc, argv, "--filter", "");
  bool sass = hasFlag(argc, argv, "--sass");
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk shaderinfo --model <dir> [--filter name] [--sass]\n"); return 2; }
  vk::Context context;
  context.setCaptureStatistics(true);
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  nr::Geometry geometry = nr::Geometry::fromValid(768, 768);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  nr::Activation* features = graph.allocate("input features", geometry.fullWidth * geometry.fullHeight, 16, nr::Format::F32);
  VkCommandBuffer commands = context.beginCommands();
  graph.record(commands, *features);
  context.endAndSubmit(commands, true);
  checkChainTimeouts(kernels);
  for (const auto& [key, pipeline] : kernels.pipelines()) {
    if (!filter.empty() && key.find(filter) == std::string::npos) continue;
    printf("== %s\n%s\n", key.c_str(), context.pipelineStatistics(pipeline, sass).c_str());
  }
  return 0;
}
}  // namespace

namespace {
int runProfile(int argc, char** argv) {
  std::string modelDir = argValue(argc, argv, "--model");
  std::string shaderDir = argValue(argc, argv, "--shaders", executableDirectory(argv[0]) + "/shaders");
  uint32_t width = (uint32_t)atoi(argValue(argc, argv, "--width", "768").c_str());
  uint32_t height = (uint32_t)atoi(argValue(argc, argv, "--height", "768").c_str());
  if (modelDir.empty()) { fprintf(stderr, "usage: dlss5vk profile --model <dir> [--width W --height H]\n"); return 2; }
  if (!getenv("DLSS5VK_CHAIN")) nr::Kernels::setChainEnabled(false);   // per-dispatch timings need the barriers
  vk::Context context;
  printf("maxComputeSharedMemorySize %u bytes\n", context.maxComputeSharedMemory());
  nr::Model model(context, modelDir, false);
  nr::Kernels kernels(context, shaderDir);
  kernels.setSiluTable(ref::siluTable());
  // Barrier/dispatch overhead: 400 trivial dispatches each followed by a full compute barrier.
  {
    nr::Activation tiny; tiny.format = nr::Format::F16; tiny.rows = 64; tiny.channels = 16; tiny.allocRows = 64;
    tiny.buffer = context.createBuffer(64 * 16 * 2, false, "tiny");
    nr::Activation tinyOut = tiny; tinyOut.format = nr::Format::E4;
    tinyOut.buffer = context.createBuffer(64 * 16, false, "tiny out");
    VkQueryPool pool = context.createTimestampPool(2);
    VkCommandBuffer commands = context.beginCommands();
    vkCmdResetQueryPool(commands, pool, 0, 2);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    for (int i = 0; i < 400; ++i) kernels.quantize(commands, tiny, tinyOut);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    context.endAndSubmit(commands, true);
    std::vector<double> stamps = context.readTimestampsMs(pool, 2);
    printf("400 trivial dispatches + barriers: %.3f ms (%.2f us each)\n", stamps[1] - stamps[0], (stamps[1] - stamps[0]) * 2.5);
    vkDestroyQueryPool(context.device(), pool, nullptr);
    context.destroyBuffer(tiny.buffer); context.destroyBuffer(tinyOut.buffer);
    context.resetDescriptorPool();
  }
  nr::Geometry geometry = nr::Geometry::fromValid(width, height);
  nr::Graph graph(context, model, kernels, geometry, {.fusedBlocks = fusedBlocksEnabled()});
  const uint32_t fullRows = geometry.fullWidth * geometry.fullHeight;
  nr::Activation* features = graph.allocate("input features", fullRows, 16, nr::Format::F32);
  std::vector<float> synthetic((size_t)fullRows * 16);
  for (size_t i = 0; i < synthetic.size(); ++i) synthetic[i] = num::roundF16(std::sin(i * 0.0017f) * 0.125f);
  context.upload(features->buffer, synthetic.data(), synthetic.size() * 4);
  {
    VkCommandBuffer commands = context.beginCommands();
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
    checkChainTimeouts(kernels);
  }
  std::map<std::string, double> byLabel;
  std::map<std::string, double> byStage;
  double total = 0;
  // Other programs share the GPU: take the per-dispatch minimum over several frames.
  const int frames = atoi(argValue(argc, argv, "--frames", "7").c_str());
  std::vector<nr::Kernels::ProfileEntry> best;
  for (int frame = 0; frame < frames; ++frame) {
    context.resetDescriptorPool();
    VkCommandBuffer commands = context.beginCommands();
    kernels.beginProfile(commands, 4096);
    graph.record(commands, *features);
    context.endAndSubmit(commands, true);
    checkChainTimeouts(kernels);
    std::vector<nr::Kernels::ProfileEntry> entries = kernels.endProfile();
    if (best.empty()) best = entries;
    for (size_t i = 0; i < entries.size() && i < best.size(); ++i)
      best[i].milliseconds = std::min(best[i].milliseconds, entries[i].milliseconds);
  }
  const std::string stageFilter = argValue(argc, argv, "--stage", "");
  for (size_t index = 0; index < best.size(); ++index) {
    const auto& entry = best[index];
    size_t bar = entry.label.find(" | ");
    std::string stage = entry.label.substr(0, bar), kernel = entry.label.substr(bar + 3);
    if (!stageFilter.empty() && stage.find(stageFilter) != std::string::npos)
      printf("  %8.2f us  [%3zu] %s | %s\n", entry.milliseconds * 1000.0, index, stage.c_str(), kernel.c_str());
    byLabel[kernel] += entry.milliseconds;
    byStage[stage] += entry.milliseconds;
    total += entry.milliseconds;
  }
  std::vector<std::pair<std::string, double>> sorted(byLabel.begin(), byLabel.end());
  std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
  printf("total %.3f ms (sum of per-dispatch spans, %ux%u)\n\nby kernel:\n", total, width, height);
  for (size_t i = 0; i < sorted.size() && i < 40; ++i)
    printf("  %8.3f ms  %5.1f%%  %s\n", sorted[i].second, 100 * sorted[i].second / total, sorted[i].first.c_str());
  std::vector<std::pair<std::string, double>> stages(byStage.begin(), byStage.end());
  std::sort(stages.begin(), stages.end(), [](auto& a, auto& b) { return a.second > b.second; });
  printf("\nby stage:\n");
  for (size_t i = 0; i < stages.size(); ++i)
    printf("  %8.3f ms  %5.1f%%  %s\n", stages[i].second, 100 * stages[i].second / total, stages[i].first.c_str());
  return 0;
}
}  // namespace

int runCommand(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "verify")) return runVerify(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "shaderinfo")) return runShaderInfo(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "profile")) return runProfile(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "parity")) return runParity(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "bench")) return runBench(argc, argv);
  fprintf(stderr, "usage: dlss5vk parity|verify|bench|profile|shaderinfo --model <dir> ...\n");
  return 2;
}

int main(int argc, char** argv) {
  int code = 1;
  try {
    code = runCommand(argc, argv);
  } catch (const std::exception& error) {
    fprintf(stderr, "error: %s\n", error.what());
  }
  // With the validation layer on, an error it reported fails the run, whatever the command concluded.
  const char* validation = getenv("DLSS5VK_VALIDATION");
  if (validation && *validation && strcmp(validation, "0")) {
    const uint32_t errors = vk::Context::validationErrors();
    printf("validation: %u error%s reported by the layer\n", errors, errors == 1 ? "" : "s");
    if (errors && code == 0) code = 1;
  }
  return code;
}
