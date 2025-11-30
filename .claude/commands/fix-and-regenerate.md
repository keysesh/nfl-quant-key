After fixing bugs in the codebase, run the complete regeneration pipeline:

1. Kill all old prediction/recommendation processes
2. Delete stale prediction files
3. Regenerate Week N predictions with fixes applied
4. Regenerate recommendations
5. Regenerate dashboard
6. Validate top 5 recommendations

Ensures consistency and prevents forgotten steps when deploying fixes.

Example usage: /fix-and-regenerate 11
