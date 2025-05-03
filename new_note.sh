#!/usr/bin/env bash
slug=$(echo "$*" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' )
fname="_random/$(date +%Y-%m-%d)-$slug.md"
printf -- "---\ntitle: \"%s\"\ndate: %s\n---\n\n" "$*" "$(date +%Y-%m-%d)" > "$fname"
$EDITOR "$fname"