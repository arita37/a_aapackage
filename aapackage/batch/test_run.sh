#!/usr/bin/env bash
dirn="$(dirname "$0")"
which_python="$(which python)"
${which_python} ${dirn}/batch_daemon_launch_cli.py --task_folder ${dirn}/ztest/tasks/ &
sleep 3
${which_python} ${dirn}/batch_daemon_monitor_cli.py --monitor_log_folder ${dirn}/ztest/monitor_logs
