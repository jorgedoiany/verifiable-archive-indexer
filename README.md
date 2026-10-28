# verifiable-archive-indexer

Indexes Spanish-language audiovisual footage and attaches to every metadata
claim the evidence that supports it, so an archivist can review what is not
verified. Built for the Meta Global AI Developer Hackathon (Indie track).

## Status

Work in progress. Nothing in this repository is functional yet; this section
will list what actually runs as each piece lands.

## Planned pipeline

1. Transcribe speech (Muse Voice Transcribe).
2. Extract metadata from video with cited evidence (Muse Spark).
3. Verify visual claims independently (SAM 3.1).
4. Queue unverified claims for human review.
5. Measure accuracy against a hand-labeled ground truth.

## Originality and scope

All code in this repository was written during the hackathon window. It uses
no data from any employer or third party; demo clips are recorded by the
author.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # then add your API key
```

## License

MIT
