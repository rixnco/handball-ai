# Prompt for Claude Code - handball-ai mission

Copy the content below when you start Claude Code at the root of the repository.

---

First read `CLAUDE.md` at the root. It holds the full mission context, the code conventions, the implemented modules and the ones to finish, and the expected git workflow.

Then read `legacy/pipeline_original.py` entirely. It is the business specification.

Finally read the project structure: `find . -type f -name "*.py" | head -30`.

Once these 3 readings are done:

1. Check that the COMPLETE modules work: run `pytest tests/ -v`. If tests fail, find out why (probably missing deps or minor bugs in my scaffolding).

2. Propose in a few lines your execution plan for steps 2 to 6 of CLAUDE.md. Which modules first, in which order, which risks you see, where you have doubts about the business specification.

3. Wait for my GO before coding.

Once I give the GO:
- Create one branch per feature (`feat/player-stats`, `feat/clipping`, etc.)
- Code a full step, commit, open a PR for this feature, wait for my approval, merge into main, move on to the next one
- At each step, SHOW ME a concrete test proving that it works
- No TODO left, no invented library APIs
- When in doubt, ask instead of inventing

When you need to test on a GPU, we will have an iteration where I run the notebook locally with a GPU and send you the logs/errors.

Start now: readings, test check, plan of attack.
