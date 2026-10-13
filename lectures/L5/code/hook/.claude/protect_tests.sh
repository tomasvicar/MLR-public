#!/bin/bash
# Claude Code PreToolUse hook (see settings.json): block Edit/Write of test_*.py.
# Reads the tool call as JSON on stdin; exit 2 blocks it and stderr goes to the model.
jq -r .tool_input.file_path | grep -q 'test_[^/]*\.py$' || exit 0
echo "Tests are read-only." >&2
exit 2
