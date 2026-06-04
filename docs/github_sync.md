# GitHub Sync

The project is intentionally standalone under:

```text
/home/ljz/llm-mas/CausalCommunicationRefiner
```

Suggested first-time setup:

```bash
cd /home/ljz/llm-mas/CausalCommunicationRefiner
git init
git add .
git commit -m "Initial causal communication refiner"
git branch -M main
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

If you prefer HTTPS:

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
```

Before pushing, check generated files are ignored:

```bash
git status --short
```

Do not commit experiment outputs under `results/`, model checkpoints, or JSONL logs.

