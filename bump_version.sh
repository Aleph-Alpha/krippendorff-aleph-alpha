#!/bin/bash

# Script to bump version in version.py
# Usage: ./bump_version.sh <new_version>
# Example: ./bump_version.sh 0.1.18

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.1.18"
    exit 1
fi

NEW_VERSION=$1
VERSION_FILE="src/krippendorff_alpha/version.py"

if [ ! -f "$VERSION_FILE" ]; then
    echo "Error: $VERSION_FILE not found"
    exit 1
fi

# Update version in version.py
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"
rm -f "${VERSION_FILE}.bak"

echo "Version updated to $NEW_VERSION in $VERSION_FILE"
echo ""
echo "Next steps:"
echo "1. Review the changes: git diff $VERSION_FILE"
echo "2. Commit: git add $VERSION_FILE && git commit -m 'Bump version to $NEW_VERSION'"
echo "3. Tag: git tag v$NEW_VERSION"
echo "4. Push: git push origin main && git push origin v$NEW_VERSION"

