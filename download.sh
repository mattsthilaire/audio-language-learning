#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: no URL provided"
    exit 1
fi

URL="$1"
USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

wget --user-agent="$USER_AGENT" "$URL"
