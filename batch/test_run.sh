#!/usr/bin/env bash
dirn="$(dirname "$0")"
which_python="$(which python)"
${which_python} ${dirn}/batch_daemon_launch_cli.py &
sleep 3
${which_python} ${dirn}/batch_daemon_monitor_cli.py --log_directory=${dirn}/monitor_logs
