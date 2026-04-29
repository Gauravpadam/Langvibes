#!/bin/bash
# Usage: ./install.sh /path/to/your/project

set -e

TARGET="${1:-.}"

if [ ! -d "$TARGET" ]; then
  echo "Error: target directory '$TARGET' does not exist."
  exit 1
fi

echo "Installing langvibes into $TARGET ..."

cp -r .claude/ "$TARGET/"
cp -r .github/ "$TARGET/"
cp -r .kiro/   "$TARGET/"
cp requirements.txt "$TARGET/"

echo "Done."
echo ""
echo "Next steps:"
echo "  1. cd $TARGET"
echo "  2. pip install -r requirements.txt"
echo "  3. aws configure  (if not already set up)"
echo "  4. Open the project in Claude Code, VS Code with Copilot, or Kiro"
