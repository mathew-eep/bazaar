#!/bin/bash
# Launch auto-retraining monitor in tmux

set -e

PROJECT_DIR="$HOME/bazaar"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"

# Ensure venv is active
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "🚀 Starting auto-retraining monitor..."
echo "   Project: $PROJECT_DIR"
echo "   Venv: $VENV_DIR"
echo ""
echo "Monitor will check for regime shifts every hour."
echo "Logs: $PROJECT_DIR/logs/monitor.log"
echo ""
echo "To view logs in real-time:"
echo "  tail -f $PROJECT_DIR/logs/monitor.log"
echo ""
echo "To stop the monitor:"
echo "  tmux kill-session -t bazaar-monitor"
echo ""

# Create or attach to tmux session
tmux new-session -d -s bazaar-monitor -x 200 -y 50

# Run monitor in the session
tmux send-keys -t bazaar-monitor "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python -m src.auto_retrain_monitor --check-interval=3600 --log-file=logs/monitor.log" Enter

# Show session
tmux ls | grep bazaar-monitor

echo "✅ Monitor launched in tmux session 'bazaar-monitor'"
echo "🔗 Attach with: tmux attach -t bazaar-monitor"
