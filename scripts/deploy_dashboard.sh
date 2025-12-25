#!/bin/bash
# Deploy NFL QUANT Dashboard to Vercel via GitHub
# Run after pipeline completes or after making style changes

set -e

PROJECT_DIR="/Users/keyonnesession/Desktop/NFL QUANT"
DEPLOY_DIR="$PROJECT_DIR/deploy"

echo "=== NFL QUANT Dashboard Deployment ==="

# Copy latest dashboard file to deploy directory
echo "Copying pro_dashboard.html..."
cp "$PROJECT_DIR/reports/pro_dashboard.html" "$DEPLOY_DIR/index.html"

# Commit and push to GitHub (triggers Vercel auto-deploy)
cd "$DEPLOY_DIR"
git add -A
git commit -m "Update dashboard $(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
git push origin main

echo ""
echo "=== Pushed to GitHub ==="
echo "Vercel will auto-deploy in ~30 seconds"
echo "View at: https://nfl-quant.vercel.app"
