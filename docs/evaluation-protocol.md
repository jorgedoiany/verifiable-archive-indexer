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

## Clarifications (before the blind run)

- Judging sheet. "Exists" means the object is clearly visible on screen at
  some moment inside the time range the model cited (one second of margin is
  allowed). "Does not exist" means it is not visible in that range, whether
  it is absent from the clip or only appears at another time, so the figures
  call these claims "not visible in the cited range", not "invented". A claim
  with no cited time range is judged against the whole clip. The sheet shows
  only the clip, the object and the range, in shuffled order.
- Proportions are reported with the counts behind them and a 95% Wilson
  interval, and also with every "unsure" counted as not visible.
- Attribute errors. Added on 2026-11-02, while the author was judging the first
  blind run and before any judging report was run or any number seen. A claim
  is judged "visible in the range" when the object it points to is visible
  there, even if a detail such as its color is wrong; the author then starts
  the note with "attr:" and says what it really is. The report counts these
  notes separately, so claims with a wrong attribute are not presented as
  fully correct. The rule appeared after the author had already judged some
  items; the author was asked to re-check the earlier ones against it, and
  which ones were re-checked is not recorded.

## Deviations

- 2026-10-31, before any pipeline run on a blind clip: the label of blind_03
  was corrected from "no speech" to a single spoken word (a first name, said
  by the author at about second 18). The author found it by listening again;
  it was not taken from any program output. The label file and the manifest
  hashes were regenerated. No blind clip had pipeline outputs at that moment
  (checked by scripts/freeze_blind.py). The previous manifest is in the Git
  history. Because that reference has one word, the word error rate is
  reported over all words pooled across clips, not averaged per clip.
- 2026-11-01, after the blind run with the basic prompt but before any of its
  results was looked at: the pipeline does not record model token counts per
  clip, so the cost reported is the SAM frames and their price. The
  transcription and extraction calls are small by comparison and are not
  itemized.
- 2026-11-02, after the blind run with the inventory prompt but before any of its
  results was looked at: 5 of the 9 blind clips completed and 4 failed with
  network read timeouts, in three different stages (transcription for one clip,
  the visual model call for two, segmentation for one). The error handling was
  fixed (network errors are now retried and reported per stage, with longer
  timeouts) and the 4 clips are run again. The partial files of the failed
  attempts are overwritten without having been looked at, and the clips that
  completed are not rerun. The reported cost covers completed runs; failed
  attempts may also have been billed and are not included.
- 2026-11-02, after judging the inventory run and before any judging report was
  generated or any number seen: all 162 object claims of the inventory run were
  judged (not a sample of 100). While judging, the author asked the assistant
  how to mark about a dozen claims, describing the claim and what was on screen;
  the assistant never saw any verifier verdict. This may have moved some claims
  between "visible with an attr: note" and "not visible". Two rules appeared
  during judging: a wrong material counts as an attribute error, like a wrong
  color; and for borderline colors (white versus beige) one criterion is chosen
  and used throughout. The precision used for the decision on the default prompt
  is the one defined above (visible over visible plus not visible, with attr:
  notes counted as visible). The strict version, counting attr: notes as errors,
  is reported next to it.

## Observations before the blind run

- With synthetic audio generated by ffmpeg (20 s of silence and 20 s of pink
  noise), the transcriber returned one word and one turn for each, although
  nobody spoke. This is reported as a limitation: invented speech can reach
  the extraction prompt as if it were evidence. The evaluation counts invented
  words separately on clips without speech.

## Decision on the default prompt (2026-11-02)

The criteria fixed above were met. With the inventory prompt, 147 of the 149
decided object claims were visible in the cited range (98.7%, 95% interval 95.2%
to 99.6%), against the 90% required, and the coverage of the labeled objects by
name rose from 21.0% (basic prompt) to 62.9%, against the 10 points required.
The inventory prompt is now the default; `--prompt basic` keeps the old one.

Caveats that go with the decision:

- 13 of the 162 inventory claims were judged "unsure", and 10 of the visible ones
  had a wrong attribute (a wrong color or material). The strict precision, which
  counts those as errors, is printed by `scripts/judge_report.py` next to the
  protocol's definition and is the figure to quote with it.
- The author consulted the assistant while judging about a dozen claims (see the
  deviation above), so some judgments may have moved.
- Only 2 claims were judged not visible, so these results cannot say whether the
  automatic checks catch invented objects. They did accept claims with a wrong
  color or material, because a segmentation match looks for the noun, not the
  adjective. For the claims where the checks found no match, 23 of the 24
  decided ones were visible, which is why a miss never discards a claim.
- Nine clips of 20 seconds from one person, with that person as the only judge,
  who also recorded and labeled the clips. Coverage depends on name matching and
  is an estimate.
