#!/bin/sh
# Build the engine-seed preload. Local gating only; nothing here ships.
set -e
here=$(cd "$(dirname "$0")" && pwd)
case "$(uname -s)" in
Darwin)
    clang++ -std=c++17 -O2 -dynamiclib -lc++ \
        -o "$here/libengine_seed.dylib" "$here/engine_seed.cpp"
    echo "built $here/libengine_seed.dylib"
    ;;
*)
    c++ -std=c++17 -O2 -shared -fPIC \
        -o "$here/libengine_seed.so" "$here/engine_seed.cpp"
    echo "built $here/libengine_seed.so"
    ;;
esac
