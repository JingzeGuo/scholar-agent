# ScholarAgent Working Agreement

## Scope

These instructions apply to the entire repository.

## Engineering rules

- Follow `CODEX_IMPLEMENTATION_PLAN.md` phase by phase and do not claim a phase is complete
  until every listed acceptance check has objective evidence.
- Keep provider-dependent tests behind the `live` pytest marker. The default test suite must
  remain deterministic and must not require paid API calls.
- Use Pydantic models at module boundaries and preserve structured, secret-free execution logs.
- Never commit `.env`, API keys, generated indexes, model caches, large PDFs, or raw provider
  responses that may contain sensitive content.
- Add or update tests for every behavior change. Before finishing, run `make quality` and
  `UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv lock --check`.

## Required commit

After code changes are complete and the relevant checks pass, commit the finished work to Git.
Use a concise commit message that describes the implemented phase or fix. Do not leave completed
code changes only in the working tree.
