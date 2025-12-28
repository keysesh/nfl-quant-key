#!/bin/bash
# Deploy NFL QUANT Dashboard to Vercel via GitHub
# Run after pipeline completes or after making style changes

set -e

PROJECT_DIR="/Users/keyonnesession/Desktop/NFL QUANT"
DEPLOY_DIR="$PROJECT_DIR/deploy"

echo "=== NFL QUANT Dashboard Deployment ==="

# Check if running Next.js or legacy mode
if [ "$1" = "--legacy" ]; then
    echo "Mode: Legacy HTML"
    # Copy latest dashboard file to deploy directory
    echo "Copying pro_dashboard.html..."
    cp "$PROJECT_DIR/reports/pro_dashboard.html" "$DEPLOY_DIR/index.html"

    # Commit and push to GitHub (triggers Vercel auto-deploy)
    cd "$DEPLOY_DIR"
    git add -A
    git commit -m "Update dashboard $(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
    git push origin main
else
    echo "Mode: Next.js"

    # Generate JSON data from latest recommendations
    echo "Generating JSON data..."
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    python scripts/dashboard/generate_pro_dashboard.py --json-only

    # Build Next.js app (Next.js is at deploy root, not web subdirectory)
    echo "Building Next.js app..."
    cd "$DEPLOY_DIR"
    npm run build

    # Commit and push to GitHub (triggers Vercel auto-deploy)
    # Only add picks.json to avoid overwriting component changes
    git add src/data/picks.json
    git commit -m "Update picks $(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
    git push origin main || echo "Push failed - may need manual intervention"
fi

echo ""
echo "=== Pushed to GitHub ==="
echo "Vercel will auto-deploy in ~30 seconds"
echo "View at: https://nfl-quant.vercel.app"
