#!/usr/bin/env sh
set -euo pipefail

# Simple installer for coding-agent-search
# Usage: curl -fsSL https://example.com/install.sh | sh [-s -- --version v0.1.0 --dest ~/.local/bin --easy-mode]

VERSION="${VERSION:-v0.1.0}"
DEST="${DEST:-$HOME/.local/bin}"
OWNER="${OWNER:-coding-agent-search}"
REPO="${REPO:-coding-agent-search}"
CHECKSUM="${CHECKSUM:-}"
CHECKSUM_URL="${CHECKSUM_URL:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
EASY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --easy-mode) EASY=1; shift;;
    --checksum) CHECKSUM="$2"; shift 2;;
    --checksum-url) CHECKSUM_URL="$2"; shift 2;;
    *) shift;;
  esac
done

mkdir -p "$DEST"
OS=$(uname -s | tr 'A-Z' 'a-z')
ARCH=$(uname -m)
# normalize arch for release names
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="arm64" ;;
esac
TAR="coding-agent-search-${VERSION}-${OS}-${ARCH}.tar.gz"
URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${TAR}"
if [ -n "$ARTIFACT_URL" ]; then
  URL="$ARTIFACT_URL"
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL" >&2
curl -fsSL "$URL" -o "$TMP/$TAR"

# Resolve checksum: explicit flag > checksum URL > default URL.sha256
if [ -z "$CHECKSUM" ]; then
  if [ -z "$CHECKSUM_URL" ]; then
    CHECKSUM_URL="${URL}.sha256"
  fi
  echo "Fetching checksum from ${CHECKSUM_URL}" >&2
  if ! CHECKSUM="$(curl -fsSL "$CHECKSUM_URL")"; then
    echo "ERROR: checksum file not found; refusing to install without verification." >&2
    exit 1
  fi
fi

echo "$CHECKSUM  $TMP/$TAR" | sha256sum -c -

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
