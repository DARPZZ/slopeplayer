#!/bin/sh
set -eu

export DISPLAY=:99
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp &
xvfb_pid=$!

# pynput connects during Python imports, so wait until the X socket exists.
attempt=0
while [ ! -S /tmp/.X11-unix/X99 ]; do
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "Xvfb exited before its display became ready" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 50 ]; then
        echo "Timed out waiting for Xvfb display :99" >&2
        exit 1
    fi
    sleep 0.1
done

exec python -u slope_rl.py "$@"
