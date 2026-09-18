#!/usr/bin/env bash
# Production entrypoint for the current SGLang 0.5.19 deployment.
set -euo pipefail
exec bash /scratch_local/user_data/yiming/serving/serve_glm53_sglang_0519.sh "$@"
