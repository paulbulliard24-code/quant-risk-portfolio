#!/usr/bin/env bash
# Run every project's test suite.
#
# Each project is self-contained with its own `src` package, so the
# suites must run in separate pytest sessions: collecting them together
# would make the two `src` packages shadow each other.
set -e
for project_dir in 0*/; do
    if [ -d "$project_dir/tests" ]; then
        echo "=== $project_dir ==="
        (cd "$project_dir" && python -m pytest)
    fi
done
