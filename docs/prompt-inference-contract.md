# Local prompt inference contract

The local enhancer runs when `DEMON_ENHANCER_PROVIDER=local`. It uses the
checkpoint directory selected by `DEMON_ENHANCER_DIR` (or the normal model
path). Checkpoint files must remain immutable while the process loads them.
Replacing files on disk does not replace the already loaded model: restart the
process to load a new checkpoint.

## Safeguards

For SA3, both greedy enhancement and sampled variations retain known instrument
and lineup constraints from their input text. The checks run before a greedy
anchor becomes a forced prefix and again after sampling. A rejected candidate
uses a valid anchor; if neither text is valid, no replacement is returned.
Known comma-separated solo cues are normalized before enhancement and variation generation: incompatible technique clauses are removed and recognized track genres are scoped to the selected instrument. Stop zero returns that normalized anchor after validation. Unaffected anchors remain unchanged. Saved prompts receive these repairs when submitted to enhancement or variations; direct audio generation is not rewritten. Different pad
positions may therefore return identical text.

The checks recognize common instrument names, solo cues, lineup sizes, known
accompaniment terms, repetition and a small set of incompatible techniques.
They are conservative lexical checks, not a general semantic classifier or a
guarantee that generated audio contains no other instruments. An unknown solo
instrument is retained verbatim. Conflicting input, such as a solo piano plus
drums, has no valid fallback. The ACE-Step arrangement path is unchanged.

Only constraints present in the supplied text can be enforced. A client that
omits an instrument or lineup cannot expect the server to infer a hidden UI
selection or the timbre of a selected LoRA.

## Randomness and revision

Fork selection and continuation use a request-owned CPU `torch.Generator`.
Neither seeds nor saved/restored global RNG state are used. Concurrent audio
sampling cannot consume the text sampler's draws, and text sampling cannot
consume or reset the audio sampler's global stream. Token ranking resolves
exact ties by token index.

Identical text, deck and coordinate are repeatable with the same checkpoint
and inference runtime. This is not a guarantee of bit-identical logits across
all hardware, Torch versions or kernel settings, nor a guarantee of identical
output audio. Changing a checkpoint or sampler intentionally changes the grid.

`GET /api/prompt-model` returns `ok`, `provider` and `revision`. For an available
local model, it also includes `checkpoint`, `sampler`, `guard` and `runtime`.
The revision hashes the loaded weights/tokenizer/configuration, sampler and
guard versions, and recorded runtime settings. Paths are not model identities.
The revision is unavailable when local inference is disabled or cannot load.

Local `/api/enhance` and `/api/variations` replies include `revision`.
Clients may pin a request with `revision=<value>`. A mismatch returns
`{"ok":false,"stale":true,"revision":"<current value>"}` without decoding.
The client should keep its current text, adopt the new identity and use it for
the next user request. A busy variation still returns HTTP 503; it is not a
revision mismatch. These endpoints must not be cached by HTTP intermediaries.

Clients should namespace refinement caches by revision, deck and exact input
text. Unversioned persisted cache entries cannot prove which model generated
them. Saved literal prompts are separate from caches and should remain intact.

An explicit local enhancement failure keeps the input (`ok:false`) instead of
silently invoking the hosted model. Explicit hosted requests remain supported
and are outside the local sampler's repeatability contract.

## Validation

Run the focused suite with the repository environment:

```sh
python -m pytest --confcutdir=tests/unit \
  tests/unit/test_prompt_constraints.py \
  tests/unit/test_prompt_guard_integration.py \
  tests/unit/test_prompt_sampler_rng.py \
  tests/unit/test_prompt_revision.py \
  tests/unit/test_prompt_variations.py \
  tests/unit/test_prompt_enhancer.py \
  tests/unit/test_prompt_loader_concurrency.py
```

The sampler test constructs a small real T5 model locally, checks that sampling
does not alter global RNG state, and runs repeated coordinates while another
thread repeatedly seeds and draws from that global state. It requires no
checkpoint download. A deployment acceptance test should additionally exercise
the intended trained checkpoint and target runtime while audio generation runs.
