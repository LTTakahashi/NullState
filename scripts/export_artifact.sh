#!/usr/bin/env bash
# Export a self-contained, de-identified artifact for double-blind review.
#
# WHY THIS EXISTS. anonymous.4open.science mirrors a whole GitHub repository.
# Pointing it at NullState would de-anonymize the author immediately: several
# tracked files legitimately carry the author's name (LICENSE, the biology
# manuscript's \author block, its PDF/DOCX builds, the pilot report). This script
# produces a SEPARATE tree containing only the identifiability study, with no
# identifying strings, ready to push to a fresh repository that the anonymizing
# mirror can safely point at.
#
#   bash scripts/export_artifact.sh [DEST]     # default: ../nullstate-artifact
#
# Then:
#   cd DEST && git init && git add -A
#   git -c user.name=Anonymous -c user.email=anon@example.invalid commit -m "Artifact"
#   # push to a NEW empty GitHub repo, then paste that repo URL into
#   # https://anonymous.4open.science and put the generated link in the paper.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$(dirname "$REPO")/nullstate-artifact}"

if [ -e "$DEST" ]; then
  echo "refusing to overwrite existing path: $DEST" >&2
  echo "remove it or pass a different destination" >&2
  exit 1
fi

echo ">> exporting to $DEST"
mkdir -p "$DEST"

# --- the study itself (tracked files only, so nothing local sneaks in) --------
( cd "$REPO" && git ls-files identifiability ) | while read -r f; do
  mkdir -p "$DEST/$(dirname "${f#identifiability/}")"
  cp "$REPO/$f" "$DEST/${f#identifiability/}"
done

# --- the figure script, repointed at the local tree ---------------------------
mkdir -p "$DEST/paper"
sed 's|^PROBE = .*|PROBE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))|' \
  "$REPO/docs/tmlr/make_figs_v3.py" > "$DEST/paper/make_figures.py"

# --- a neutral licence (the repo LICENSE names the copyright holder) ----------
cat > "$DEST/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 Anonymous Author(s)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# --- strip anything that could identify the author ----------------------------
# absolute home paths appear in a few docstrings//headers
{ grep -rlI "/home/" "$DEST" 2>/dev/null || true; } | while read -r f; do
  [ -n "$f" ] && sed -i 's|/home/[A-Za-z0-9_.-]*/|/path/to/|g' "$f"
done
# the parent project's name is not identifying on its own, but the GitHub slug is
{ grep -rlI "github.com/" "$DEST" 2>/dev/null || true; } | while read -r f; do
  [ -n "$f" ] && sed -i 's|github\.com/[A-Za-z0-9_.-]*/[A-Za-z0-9_.-]*|github.com/ANONYMIZED|g' "$f"
done

cat > "$DEST/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
*.log
results_*_s[0-9]*.csv
EOF

# --- verification: fail loudly rather than ship a leak ------------------------
echo ">> scanning for identifying strings"
PATTERN='takahashi|luiz|voiland|washington state|wsu\.edu|ltaktakahashi|/home/[a-z]'
HITS="$(grep -rilIE "$PATTERN" "$DEST" 2>/dev/null || true)"
if [ -n "$HITS" ]; then
  echo "!! LEAK FOUND -- do not publish:" >&2
  echo "$HITS" >&2
  exit 1
fi

N=$(find "$DEST" -type f | wc -l)
echo ">> clean: no identifying strings found in $N files"
echo ">> NOTE: PDFs/binaries are not text-scanned; this export contains none by design."
echo ">> next: cd $DEST && git init && git add -A && git commit"
