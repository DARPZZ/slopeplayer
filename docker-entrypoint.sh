#!/bin/sh
set -eu

# Container restarts retain /tmp, so a crashed Xvfb can leave its display lock
# behind. Pick the first unused display instead of repeatedly failing on :99.
display_number=99
while [ -e "/tmp/.X${display_number}-lock" ] \
    || [ -S "/tmp/.X11-unix/X${display_number}" ]; do
    display_number=$((display_number + 1))
    if [ "$display_number" -gt 199 ]; then
        echo "Could not find a free X display between :99 and :199" >&2
        exit 1
    fi
done

export DISPLAY=":${display_number}"
Xvfb "$DISPLAY" -screen 0 1280x720x24 -nolisten tcp &
xvfb_pid=$!

# pynput connects during Python imports, so wait until the X socket exists.
attempt=0
while [ ! -S "/tmp/.X11-unix/X${display_number}" ]; do
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "Xvfb exited before its display became ready" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 50 ]; then
        echo "Timed out waiting for Xvfb display $DISPLAY" >&2
        exit 1
    fi
    sleep 0.1
done

exec python -u slope_rl.py "$@"
