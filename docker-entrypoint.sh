#!/bin/sh
set -eu

# Compose uses this to continue an interrupted run without making a first-time
# container fail on a missing checkpoint. Explicit --resume or --overwrite
# always wins over this convenience behavior.
if [ "${SLOPE_AUTO_RESUME:-0}" = "1" ] && [ "${1:-}" = "train" ]; then
    model_path="/app/runs/slope_qrdqn"
    explicit_mode=0
    expect_model=0
    for argument in "$@"; do
        if [ "$expect_model" = "1" ]; then
            model_path="$argument"
            expect_model=0
            continue
        fi
        case "$argument" in
            --model)
                expect_model=1
                ;;
            --model=*)
                model_path=${argument#--model=}
                ;;
            --resume|--overwrite)
                explicit_mode=1
                ;;
        esac
    done

    case "$model_path" in
        *.zip) checkpoint_base=${model_path%.zip} ;;
        *) checkpoint_base=$model_path ;;
    esac
    model_file="${checkpoint_base}.zip"
    replay_file="${checkpoint_base}.replay.pkl"
    model_exists=0
    replay_exists=0
    [ -f "$model_file" ] && model_exists=1
    [ -f "$replay_file" ] && replay_exists=1

    if [ "$explicit_mode" = "0" ]; then
        if [ "$model_exists" = "1" ] && [ "$replay_exists" = "1" ]; then
            echo "Found a complete checkpoint; resuming $checkpoint_base" >&2
            set -- "$@" --resume
        elif [ "$model_exists" != "$replay_exists" ]; then
            echo "Incomplete checkpoint: both $model_file and $replay_file are required." >&2
            exit 2
        fi
    fi
fi

# Run Python as a child so SIGTERM from `docker stop` can become SIGINT. The
# training command handles SIGINT like Ctrl+C and saves a resumable checkpoint.
python -u slope.py "$@" &
child_pid=$!

forward_stop() {
    if kill -0 "$child_pid" 2>/dev/null; then
        echo "Stop requested; asking the trainer to save its checkpoint..." >&2
        kill -INT "$child_pid"
    fi
}

trap forward_stop INT TERM
status=0
while :; do
    if wait "$child_pid"; then
        status=0
    else
        status=$?
    fi
    # A signal can interrupt `wait` while Python is still serializing the replay
    # buffer. Keep PID 1 alive until the trainer actually finishes saving.
    if ! kill -0 "$child_pid" 2>/dev/null; then
        break
    fi
done
exit "$status"
