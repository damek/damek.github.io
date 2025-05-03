#!/usr/bin/env bash
slug=$(echo "$*" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' )

# ── minimal change: filename no longer carries the date ──
fname="_random/${slug}.md"

printf -- "---\ntitle: \"%s\"\ndate: %s\n---\n\n" "$*" "$(date +%Y-%m-%d)" > "$fname"
$EDITOR "$fname"
