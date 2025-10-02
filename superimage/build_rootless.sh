#!/bin/bash
#
# Build script for rootless Apptainer image
# This script handles the fakeroot permission issues
#

set -e

# Default values
DEF_FILE="apptainer_rootless_v2.def"
OUTPUT_DIR="/share/j_sun/xy468/dts_agent/aira-dojo/superimage_rootless"
VERSION="2025-05-02v2"
OUTPUT_FILE="$OUTPUT_DIR/superimage.root.$VERSION.sif"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--def-file)
            DEF_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            OUTPUT_FILE="$OUTPUT_DIR/superimage.root.$VERSION.sif"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --def-file FILE    Definition file to use (default: $DEF_FILE)"
            echo "  -o, --output FILE      Output SIF file path"
            echo "  -v, --version VERSION  Version string for default output naming"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Building Apptainer image..."
echo "Definition file: $DEF_FILE"
echo "Output file: $OUTPUT_FILE"

# Try different build approaches in order of preference
echo "Attempting build with --ignore-fakeroot-command..."
if apptainer build --ignore-fakeroot-command "$OUTPUT_FILE" "$DEF_FILE"; then
    echo "✅ Build successful with --ignore-fakeroot-command!"
    exit 0
fi

echo "❌ Build with --ignore-fakeroot-command failed."
echo "Attempting build with --sandbox (for debugging)..."
SANDBOX_DIR="${OUTPUT_FILE%.sif}_sandbox"
if apptainer build --sandbox --ignore-fakeroot-command "$SANDBOX_DIR" "$DEF_FILE"; then
    echo "✅ Sandbox build successful!"
    echo "Converting sandbox to SIF..."
    if apptainer build "$OUTPUT_FILE" "$SANDBOX_DIR"; then
        echo "✅ SIF conversion successful!"
        rm -rf "$SANDBOX_DIR"
        exit 0
    else
        echo "❌ SIF conversion failed. Sandbox available at: $SANDBOX_DIR"
        exit 1
    fi
fi

echo "❌ All build attempts failed."
echo ""
echo "Troubleshooting suggestions:"
echo "1. Check if you have user namespaces enabled:"
echo "   cat /proc/sys/user/max_user_namespaces"
echo ""
echo "2. Try building on a different system with proper user namespace support"
echo ""
echo "3. Use the pre-built base image approach (see apptainer_minimal.def)"
echo ""
echo "4. Contact your system administrator about enabling user namespaces"

exit 1

