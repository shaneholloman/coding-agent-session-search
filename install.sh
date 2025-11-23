#!/usr/bin/env sh
set -euo pipefail

# Simple installer for coding-agent-search
# Usage: curl -fsSL https://example.com/install.sh | sh [-s -- --version v0.1.0 --dest ~/.local/bin --easy-mode]

VERSION="${VERSION:-v0.1.0}"
DEST="${DEST:-$HOME/.local/bin}"
OWNER="${OWNER:-coding-agent-search}"
REPO="${REPO:-coding-agent-search}"
CHECKSUM="${CHECKSUM:-}"
EASY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --easy-mode) EASY=1; shift;;
    *) shift;;
  esac
done

mkdir -p "$DEST"
OS=$(uname -s | tr 'A-Z' 'a-z')
ARCH=$(uname -m)
TAR="coding-agent-search-${VERSION}-${OS}-${ARCH}.tar.gz"
URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${TAR}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL" >&2
curl -fsSL "$URL" -o "$TMP/$TAR"

if [ -n "$CHECKSUM" ]; then
  echo "$CHECKSUM  $TMP/$TAR" | sha256sum -c -
fi

echo "Extracting" >&2
tar -xzf "$TMP/$TAR" -C "$TMP"

BIN="$TMP/coding-agent-search"
install -m 0755 "$BIN" "$DEST"

echo "Installed to $DEST/coding-agent-search"
if [ $EASY -eq 1 ]; then
  case :$PATH: in
    *:$DEST:*) :;;
    *) echo "Add $DEST to PATH";;
  esac
fi
