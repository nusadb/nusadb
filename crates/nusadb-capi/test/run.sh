#!/usr/bin/env bash
# Build the C ABI shared library, compile the C integration test, boot a real nusadb-server, and
# run the test against it. Honours CARGO_TARGET_DIR; works with any C compiler (the test loads the
# library at runtime rather than linking it).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
profile=debug

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) dll="nusadb_capi.dll"; exe="capi_test.exe"; server="nusadb-server.exe" ;;
  Darwin) dll="libnusadb_capi.dylib"; exe="capi_test"; server="nusadb-server" ;;
  *) dll="libnusadb_capi.so"; exe="capi_test"; server="nusadb-server" ;;
esac

echo "==> building the C ABI library + server"
( cd "$repo_root" && cargo build -p nusadb-capi -p nusadb-server )

lib_path="$target_dir/$profile/$dll"
server_path="$target_dir/$profile/$server"
[ -f "$lib_path" ] || { echo "library not found: $lib_path" >&2; exit 1; }

echo "==> compiling the C test"
out_dir="$target_dir/$profile"
cc="${CC:-gcc}"
"$cc" -O2 -Wall -o "$out_dir/$exe" "$here/test.c" ${DLOPEN_LIBS:-}

echo "==> starting nusadb-server"
port=$(( (RANDOM % 20000) + 20000 ))
data_dir="$(mktemp -d)"
RUST_LOG=error "$server_path" --listen "127.0.0.1:$port" --data-dir "$data_dir" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; rm -rf "$data_dir"' EXIT

# Wait for readiness.
for _ in $(seq 1 150); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then exec 3>&- 3<&-; break; fi
  sleep 0.1
done

echo "==> running the C test"
"$out_dir/$exe" "$lib_path" 127.0.0.1 "$port"
