---
trigger: always_on
---

# Project Git Commit Rule (Strict)

1. **One File Per Commit**:
   - Stage and commit each modified or added file **separately** (1 file = 1 commit).
   - Never commit or push `git commit.md`.

2. **Alternate GitHub Identities on Every Commit**:
   - Strictly alternate author and committer identities on each consecutive commit between the two accounts below (check `git log -1 --format="%ae"` before committing to pick the opposite account):

   **Account A — Tahsin Raihan Robbani:**
   ```powershell
   git -c user.name="Tahsin Raihan Robbani" -c user.email="tahsinraihanrobbani@gmail.com" commit -m "Commit Message"
   ```

   **Account B — Hasin Ishrak:**
   ```powershell
   git -c user.name="Hasin Ishrak" -c user.email="hasinishrak2015@gmail.com" commit -m "Commit Message"
   ```

   *(If `GIT_AUTHOR_*` / `GIT_COMMITTER_*` env vars are set in the shell, override or unset them first so `-c user.name` and `-c user.email` set both Author and Committer.)*
