# Pinned submodule versions

These are the commits referenced by this repository’s `git submodule` entries. Update this file when you bump submodules for a paper revision.

| Submodule | Branch | Commit |
|-----------|--------|--------|
| [third_party/sglang-fast-rotation](third_party/sglang-fast-rotation) | `colm_rotation_fast` | `0fcc241961f9c79c27f6bad9a456bf10c8554a84` |
| [third_party/sglang-kmeans](third_party/sglang-kmeans) | `jinda_kmeans_rotation_dump` | `43925c00fb91ce58eb2d9c6836bb2f9885ff618f` |

Both submodules use the upstream fork: [github.com/jindajia/sglang-fork](https://github.com/jindajia/sglang-fork).

## Nested submodules inside forks (optional)

Some fork branches may list extra nested submodules (for example historical eval tooling). **Paper accuracy reproduction uses only [openai/simple-evals](https://github.com/openai/simple-evals)** against SGLang’s OpenAI-compatible API, as described in [README.md](README.md#prepare). You do not need any private nested submodule for that path. If `git submodule update --init --recursive` fails, run `git submodule update --init third_party/sglang-fast-rotation third_party/sglang-kmeans` or [scripts/clone_submodules.sh](scripts/clone_submodules.sh).
