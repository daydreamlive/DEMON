#!/usr/bin/env bash
# Pull the latest demos/pwa tree from the rtmg repo's hth-demo branch
# and install it into /root/ace-step/demos/pwa on the Vast instance.
#
# Uses the acestep_deploy SSH key already copied to /root/.ssh/.
# Designed to run ON THE INSTANCE (pushed there via scp first).

set -euo pipefail

KEY=/root/.ssh/acestep_deploy
REPO=git@github.com:ryanontheinside/rtmg.git
BRANCH=hth-demo
DEST=/root/ace-step/demos/pwa
TMP=/tmp/rtmg-$(date +%s)

export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

echo "[pull_pwa] cloning $REPO#$BRANCH -> $TMP"
git clone --depth 1 --branch "$BRANCH" "$REPO" "$TMP"

cd "$TMP"
echo "[pull_pwa] head:"
git log -1 --oneline

echo "[pull_pwa] files in demos/pwa:"
ls -la demos/pwa

echo "[pull_pwa] syncing demos/pwa -> $DEST"
mkdir -p "$DEST"
rsync -a --delete demos/pwa/ "$DEST/"

echo "[pull_pwa] destination contents:"
ls -la "$DEST"

# Cleanup the shallow clone.  Keep the deploy key in place for future
# pulls (the user uploaded it; only they should remove it).
rm -rf "$TMP"

echo "[pull_pwa] DONE"
