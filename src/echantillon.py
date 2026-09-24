from pathlib import Path
raw = Path("data/raw"); out = Path("data/echantillon"); out.mkdir(parents=True, exist_ok=True)
for f in sorted(raw.glob("*.txt")):
    print(f.name, round(f.stat().st_size / 1e6), "Mo")
    with open(f, encoding="utf-8", errors="replace") as src, open(out / f.name, "w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            if i >= 2000: break
            dst.write(line)