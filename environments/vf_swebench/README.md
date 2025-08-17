# vf-swebench

### Overview
- **Environment ID**: `vf-swebench`
- **Short description**: Single-turn SWE-bench style tasks; the model outputs a unified diff patch inside XML tags; graded by structure and similarity.
- **Tags**: code, patch, single-turn, xml, swebench

### Quickstart

```bash
uv run vf-eval vf-swebench
```

With custom args (limit rows, pick a split):

```bash
uv run vf-eval vf-swebench \
  -a '{"hf_dataset_path": "princeton-nlp/SWE-bench_Lite", "split": "train", "limit": 100}'
```

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-swebench -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->