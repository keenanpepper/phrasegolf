#!/bin/bash
# Cron script to pre-compute hints daily
# Schedule this to run daily at 2:30am Pacific (before 3am game change)
# 
# To install in crontab (for 2:30am Pacific):
#   30 2 * * * /path/to/phrasegolf/cron-precompute.sh >> /path/to/phrasegolf/precompute.log 2>&1

# Change to script directory
cd "$(dirname "$0")"

echo "=== Hint pre-computation started at $(date) ==="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the precompute script for today and the next 2 days
python precompute_hints.py --days 3

echo "=== Hint pre-computation finished at $(date) ==="
echo ""

