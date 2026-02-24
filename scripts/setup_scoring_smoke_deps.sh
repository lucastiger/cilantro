#!/usr/bin/env bash
set -euo pipefail

# Download MHCflurry model bundles needed by smoke tests.
mhcflurry-downloads fetch models_class1_pan models_class1_presentation

echo "MHCflurry smoke-test assets downloaded."
