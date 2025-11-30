# Hint Pre-computation Setup

## Overview

The hint feature requires pre-computing similarity scores for ~16.5k phrases against each day's target. This takes several minutes, so we pre-compute these hints before deploying and daily before the game changes at **3am Pacific**.

## Files

- `precompute_hints.py` - Main script to compute hints
- `cron-precompute.sh` - Cron wrapper script for daily automation
- `hints_cache/` - Directory where computed hints are stored

## Manual Usage

### Compute hints for today and tomorrow:
```bash
python precompute_hints.py
```

### Compute hints for the next 7 days:
```bash
python precompute_hints.py --days 7
```

### Compute hints for a specific game number:
```bash
python precompute_hints.py --game 150
```

## Deployment Setup

### For AWS Elastic Beanstalk (or similar):

1. **Before deploying**, pre-compute hints locally:
   ```bash
   python precompute_hints.py --days 3
   ```

2. **Include the `hints_cache/` directory** in your deployment package:
   - Make sure `hints_cache/` with the JSON files is included in your zip
   - The cache files will be available immediately when the app starts

3. **Set up daily automation on the server**:
   
   Create a cron job that runs at 2:30am Pacific daily:
   ```bash
   crontab -e
   ```
   
   Add this line (adjust paths as needed):
   ```
   30 2 * * * cd /var/app/current && /var/app/current/cron-precompute.sh >> /var/app/current/precompute.log 2>&1
   ```

   Or for AWS EventBridge/Lambda alternative, trigger the script at 2:30am Pacific daily.

## Notes

- The game day changes at **3am Pacific** (the code subtracts 3 hours from current time)
- Pre-computing one day's hints takes approximately 5-10 minutes
- Each cache file is around 500KB-1MB
- Cache files are stored as: `hints_cache/game_{NUMBER}.json`
- The application will automatically use cached hints if available
- If a cache is missing, the app will compute on-demand (but this will cause timeouts on AWS)

## Monitoring

Check the pre-computation log:
```bash
tail -f precompute.log
```

List available cached games:
```bash
ls -lh hints_cache/
```

## Initial Setup for Production

Before first deployment to production:

1. Pre-compute hints for the next week:
   ```bash
   python precompute_hints.py --days 7
   ```

2. Include `hints_cache/` in your deployment

3. Set up the cron job to keep generating new hints daily

4. The system will have hints ready for users immediately!

