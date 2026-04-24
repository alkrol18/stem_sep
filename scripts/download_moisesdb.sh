#!/usr/bin/env bash
# MoisesDB requires manual download via a form submission.
# Automatic scraping or bypassing the form is NOT supported.
#
# Steps:
#   1. Visit https://moises.ai/research/moisesdb/
#   2. Fill out the access request form (academic / research use).
#   3. After approval you will receive a download link or instructions by email.
#   4. Download the dataset and extract it to:
#        ./datasets/moisesdb/
#      The expected structure is:
#        datasets/moisesdb/
#        ├── <track_id>/
#        │   ├── mixture.wav  (optional, may be absent)
#        │   └── <stem_name>.wav   (one file per stem)
#        └── ...
#   5. Run the index builder:
#        python scripts/prepare_moisesdb.py --root ./datasets/moisesdb
echo "MoisesDB must be downloaded manually. See comments in this file."
