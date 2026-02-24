#!/usr/bin/env bash
#
# build-rpm.sh — Build an RPM package for Genieus
#
# The RPM bundles the Docker image and a systemd unit file.
# On install it loads the image, enables and starts the service.
#
# Prerequisites: docker, rpmbuild (rpm-build package)
#
# Usage:
#   ./packaging/build-rpm.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="1.0.0"
IMAGE_NAME="genieus"
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"
IMAGE_TAR="${IMAGE_NAME}-${VERSION}.tar.gz"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${CYAN}→${RESET} $*"; }
ok()    { echo -e "${GREEN}✓${RESET} $*"; }
err()   { echo -e "${RED}✗${RESET} $*" >&2; }

# ── Pre-flight checks ──────────────────────────────────────────────
for cmd in docker rpmbuild; do
    if ! command -v "$cmd" &>/dev/null; then
        err "'$cmd' is required but not found."
        echo "  Install with: sudo dnf install rpm-build   (or yum/zypper equivalent)"
        exit 1
    fi
done

# ── Step 1: Build Docker image ─────────────────────────────────────
info "Building Docker image ${BOLD}${IMAGE_TAG}${RESET} ..."
docker build -t "$IMAGE_TAG" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"
ok "Docker image built"

# ── Step 2: Export image to tarball ─────────────────────────────────
info "Exporting image to tarball ..."
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

docker save "$IMAGE_TAG" | gzip > "$TMPDIR/$IMAGE_TAR"
IMAGE_SIZE="$(du -h "$TMPDIR/$IMAGE_TAR" | cut -f1)"
ok "Image exported (${IMAGE_SIZE})"

# ── Step 3: Set up rpmbuild tree ────────────────────────────────────
RPMBUILD_DIR="$TMPDIR/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

cp "$TMPDIR/$IMAGE_TAR"              "$RPMBUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/genieus.service"     "$RPMBUILD_DIR/SOURCES/"
cp "$SCRIPT_DIR/genieus.spec"        "$RPMBUILD_DIR/SPECS/"

# ── Step 4: Build RPM ──────────────────────────────────────────────
info "Building RPM package ..."
rpmbuild \
    --define "_topdir $RPMBUILD_DIR" \
    -bb "$RPMBUILD_DIR/SPECS/genieus.spec"

# ── Step 5: Copy RPM to output ─────────────────────────────────────
OUTPUT_DIR="$PROJECT_ROOT/dist"
mkdir -p "$OUTPUT_DIR"
find "$RPMBUILD_DIR/RPMS" -name '*.rpm' -exec cp {} "$OUTPUT_DIR/" \;

RPM_FILE="$(ls "$OUTPUT_DIR"/${IMAGE_NAME}-*.rpm 2>/dev/null | head -1)"

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  RPM package built successfully!${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${CYAN}Package:${RESET}  $RPM_FILE"
echo -e "  ${CYAN}Size:${RESET}     $(du -h "$RPM_FILE" | cut -f1)"
echo ""
echo -e "  ${BOLD}Install:${RESET}  sudo rpm -ivh $RPM_FILE"
echo -e "  ${BOLD}Remove:${RESET}   sudo rpm -e genieus"
echo ""
