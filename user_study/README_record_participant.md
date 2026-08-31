# Participant feedback recorder

Place `record_participant.py` inside your existing project `user_study` folder. It is a separate study-data helper and should **not** be added to the frozen participant-facing manifest.

## Condition A

```powershell
python .\rq3b_user_eval_main.py --condition vap_turngpt --participant P01
python .\user_study\record_participant.py --participant P01 --condition A
```

The recorder looks for `rq3_results/rq3b_vap_turngpt_P01_*.csv` and asks you to confirm the detected raw log.

## Condition B

```powershell
python .\main.py
python .\user_study\record_participant.py --participant P02 --condition B
```

Because frozen Condition B does not embed the participant ID in its raw filename, the recorder selects the newest `logs/*_turntaking_*.csv` and asks you to confirm it.

If necessary, provide a raw log explicitly:

```powershell
python .\user_study\record_participant.py --participant P02 --condition B --log ".\logs\Pxxxx_turntaking_xxx.csv"
```

## Output

- `user_study/data/study_master.csv` — one analysis-ready row per participant.
- `user_study/data/P01/participant_record.json` — full participant-level record.
- Raw robot logs are never renamed or modified.
- The raw-log SHA-256 is stored so the mapping can be audited later.
- Q5 is preserved as entered and its reverse-scored value is stored separately.

## Duplicate protection

The script refuses to create a second record for the same participant ID. For an intentional correction only:

```powershell
python .\user_study\record_participant.py --participant P01 --condition A --overwrite
```

`study_master.csv` is backed up to `study_master.backup.csv` before a rewrite.
