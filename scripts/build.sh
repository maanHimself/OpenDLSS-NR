#!/usr/bin/env bash
# Linux counterpart of build.ps1: compile the shaders and PTX kernels (build_shaders.sh), then build/dlss5vk with
# GCC or Clang. Run scripts/fetch_tools.sh first. --debug builds without optimisation and with symbols.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
"$root/scripts/build_shaders.sh"

cxx="${CXX:-g++}"
cc="${CC:-gcc}"
opt="-O2"
[ "${1:-}" = "--debug" ] && opt="-O0 -g"
out="$root/build"
mkdir -p "$out/obj"
include=(-I"$root/tools/Vulkan-Headers/include" -I"$root/tools/volk" -I"$root/src")
"$cc" $opt -DVK_ENABLE_BETA_EXTENSIONS "${include[@]}" -c "$root/tools/volk/volk.c" -o "$out/obj/volk.o"
objects=("$out/obj/volk.o")
for source in "$root"/src/*.cpp; do
  object="$out/obj/$(basename "${source%.cpp}").o"
  "$cxx" -std=c++20 $opt -DVK_ENABLE_BETA_EXTENSIONS "${include[@]}" -c "$source" -o "$object" &
  objects+=("$object")
done
wait_all() { local failed=0; for job in $(jobs -p); do wait "$job" || failed=1; done; return $failed; }
wait_all || { echo "compilation failed" >&2; exit 1; }
"$cxx" "${objects[@]}" -o "$out/dlss5vk" -ldl -lpthread
echo "built $out/dlss5vk"
