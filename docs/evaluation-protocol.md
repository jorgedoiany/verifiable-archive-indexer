# Evaluation protocol

Written and committed before the pipeline was run on any blind clip. Anything
that changes after this commit is listed under "Deviations" at the end; nothing
is changed silently.

## Data

- 9 blind clips of 20 seconds each (blind_01 to blind_08 and blind_10), all from
  one person's own phone gallery: one speaker, Dominican Spanish or no speech,
  home and street scenes. One clip has the author playing a piano.
- Selection: a pool of 15 clips was prepared with the same rules (first 20
  seconds, metadata stripped) and split once, with a seeded shuffle (seed
  20261031), into 5 tuning clips and 10 blind clips. The split is never redrawn.
- blind_09 was excluded before anything was run because other people appear in
  it (privacy). That leaves 9 blind clips.
- Labels were written by the author by watching each clip, before anything was
  run on it. The SHA-256 of every blind clip and label file is in
  docs/blind-manifest.json, and "python scripts/freeze_blind.py --verify" checks
  that nothing changed.
- The clips and labels are personal and are not published. Only aggregate
  numbers are.

## Limits of the sample (stated up front)

- One person, one phone, one accent. Nothing here says how the system behaves
  with several speakers, other languages or professional footage.
- Nine clips of 20 seconds is small. Percentages are reported with the counts
  behind them.

## Variants

Each variant is run once per blind clip, with the same models: Muse Voice
Transcribe, Muse Spark 1.3 and SAM 3.1.

- basic: the original extraction prompt.
- inventory: an exhaustive prompt (every clearly visible object, up to 25, one
  claim per appearance).

## Measurements

1. Transcription: word error rate on clips with speech; words invented on clips
   without speech.
2. Human judgment: the author judges every object claim as exists, does not
   exist or unsure, from a shuffled sheet that shows only the claim and its time
   range (no verdict of any verifier). Unsure is reported separately.
   - Precision of object claims = exists / (exists + does not exist).
   - The same figure for the claims SAM matched, and for the claims SAM did not
     match (how many of those actually exist).
3. Coverage: the share of labeled objects matched by at least one claim, using
   the frozen name lists.
4. Cost: SAM frames and model tokens per clip.

## Decisions fixed now

- The inventory prompt becomes the default only if the precision of its object
  claims is at least 90% and its coverage is at least 10 percentage points above
  the basic prompt. Otherwise the basic prompt stays the default.
- Whatever the numbers, the interface and the README say "automatic checks
  agree", never "verified" or "correct": both a SAM match and a second-model
  match were seen to accept objects that are not there.
- A SAM "no match" never discards a claim; it sends it to human review.
- All results are reported, including bad ones. No clip is dropped after the
  run.

## Deviations

None yet.
