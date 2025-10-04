#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Script to clean up orphaned Apptainer instances
# This can be run manually if instances get stuck

set -e

echo "Cleaning up orphaned Apptainer instances..."

# List all instances
echo "Current instances:"
apptainer instance list -a

# Find and stop sandbox instances that match our pattern
SANDBOX_INSTANCES=$(apptainer instance list -a | grep -E "sandbox-[a-zA-Z0-9]+" | awk '{print $1}' || true)

if [ -z "$SANDBOX_INSTANCES" ]; then
    echo "No sandbox instances found to clean up."
else
    echo "Found sandbox instances to clean up:"
    echo "$SANDBOX_INSTANCES"
    
    for instance in $SANDBOX_INSTANCES; do
        echo "Stopping instance: $instance"
        apptainer instance stop "$instance" || {
            echo "Failed to stop $instance gracefully, trying force..."
            apptainer instance stop --force "$instance" || {
                echo "Warning: Could not stop instance $instance"
            }
        }
    done
fi

# Clean up any leftover temporary directories in scratch space
if [[ -n "$SLURM_JOBID" && -d "/scratch/$SLURM_JOBID/" ]]; then
    echo "Cleaning up temporary directories in scratch space..."
    find "/scratch/$SLURM_JOBID/" -name "sandbox-*" -type d -exec rm -rf {} + 2>/dev/null || true
fi

echo "Cleanup completed."
echo "Final instance list:"
apptainer instance list -a
