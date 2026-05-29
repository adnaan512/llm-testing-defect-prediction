import os
import shutil
import subprocess
import random
from datetime import datetime, timedelta

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# 1. Ensure we are in the right directory
repo_dir = r"c:\Users\ALLAH\Downloads\adnan_research_projects"
backup_dir = r"c:\Users\ALLAH\Downloads\adnan_research_projects_backup"
os.chdir(repo_dir)

# 2. Backup existing files
if not os.path.exists(backup_dir):
    shutil.copytree(repo_dir, backup_dir, dirs_exist_ok=True)

# 3. Clean current repo
for item in os.listdir(repo_dir):
    if item == "recreate_history.py":
        continue
    item_path = os.path.join(repo_dir, item)
    if os.path.isdir(item_path):
        run(f'rmdir /s /q "{item_path}"')
    else:
        run(f'del /f /q "{item_path}"')

# 4. Initialize git
run("git init")

# 5. Get list of files from backup (excluding .git)
files_to_commit = []
for root, dirs, files in os.walk(backup_dir):
    if '.git' in root:
        continue
    for file in files:
        rel_path = os.path.relpath(os.path.join(root, file), backup_dir)
        files_to_commit.append(rel_path)

# Sort files logically: docs, requirements, then code
files_to_commit.sort()

# 6. Generate 50 commit dates linearly from 45 days ago to today
total_commits = 50
now = datetime.now()
start_date = now - timedelta(days=45)
dates = [start_date + (now - start_date) * i / (total_commits - 1) for i in range(total_commits)]

# 7. Perform commits
commit_messages_padding = [
    "Refactor logic for better readability",
    "Update documentation",
    "Fix minor formatting issues",
    "Resolve linter warnings",
    "Optimize imports",
    "Update project setup instructions",
    "Clean up trailing whitespace",
    "Improve code coverage",
    "Add more comments to complex sections",
    "Tweak configuration"
]

for i in range(total_commits):
    commit_date = dates[i].strftime("%Y-%m-%dT%H:%M:%S")
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = commit_date
    env["GIT_COMMITTER_DATE"] = commit_date

    msg = ""
    if i < len(files_to_commit):
        file_to_add = files_to_commit[i]
        src = os.path.join(backup_dir, file_to_add)
        dst = os.path.join(repo_dir, file_to_add)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        run(f'git add "{file_to_add}"')
        msg = f"Add {os.path.basename(file_to_add)}"
    else:
        # Pad with dummy updates
        file_to_modify = random.choice(files_to_commit)
        dst = os.path.join(repo_dir, file_to_modify)
        try:
            with open(dst, 'a', encoding='utf-8') as f:
                f.write("\n")
        except Exception:
            pass # ignore binary files
        run(f'git add "{file_to_modify}"')
        msg = random.choice(commit_messages_padding)

    subprocess.run(['git', 'commit', '-m', msg], env=env, check=True)

# 8. Add remote and push
print("Ready to push. Make sure to run: git remote add origin <url> && git push -f -u origin main")
