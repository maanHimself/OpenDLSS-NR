#!/usr/bin/env bash
# Linux counterpart of fetch_tools.ps1 for the command-line tool: glslang (release zip), Vulkan-Headers and volk
# (tagged clones) into tools/ (git-ignored), at the same versions. The demo's CMake, Ninja, Filament and npm
# modules are not fetched.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tools="$root/tools"
glslangVersion="16.6.0"
vulkanHeadersTag="v1.4.363"
volkTag="vulkan-sdk-1.4.357.0"
mkdir -p "$tools"

# glslang: the Linux x64 release zip (bin/glslang)
if [ ! -x "$tools/glslang/bin/glslang" ]; then
  stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' EXIT
  curl -fsSL -o "$stage/glslang.zip" \
    "https://github.com/KhronosGroup/glslang/releases/download/$glslangVersion/glslang-$glslangVersion-linux-x86_64-release.zip"
  mkdir -p "$tools/glslang"
  (cd "$tools/glslang" && unzip -q "$stage/glslang.zip")
  [ -x "$tools/glslang/bin/glslang" ] || { echo "bin/glslang not found in the glslang zip" >&2; exit 1; }
fi
"$tools/glslang/bin/glslang" --version | head -n 1

# Vulkan-Headers and volk: shallow tagged clones (only the include tree and volk.c / volk.h are used)
[ -f "$tools/Vulkan-Headers/include/vulkan/vulkan.h" ] ||
  git clone --depth 1 --branch "$vulkanHeadersTag" https://github.com/KhronosGroup/Vulkan-Headers.git "$tools/Vulkan-Headers"
[ -f "$tools/volk/volk.c" ] ||
  git clone --depth 1 --branch "$volkTag" https://github.com/zeux/volk.git "$tools/volk"
grep -m1 '#define VK_HEADER_VERSION ' "$tools/Vulkan-Headers/include/vulkan/vulkan_core.h"
