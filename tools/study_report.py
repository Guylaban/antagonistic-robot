"""Write a study report (report.md and report.json) from a CRAB session database.

    python tools/study_report.py data/sessions.db
    python tools/study_report.py data/sessions.db --participants P01 P02 --title "Pilot"

The JSON holds every number; the Markdown fills docs/reporting_template.md (or --template)
with them. Each run writes to a new folder (reports/<timestamp>/ by default) and never
overwrites an earlier report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from antagonist_robot.logging.study_report import build_report, render_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="session database (logging.db_path in config.yaml)")
    ap.add_argument("--sessions", nargs="*", help="only these session IDs")
    ap.add_argument("--participants", nargs="*", help="only these participant IDs")
    ap.add_argument("--title", default="CRAB study report")
    ap.add_argument("--template", help="Markdown template (default docs/reporting_template.md)")
    ap.add_argument("--out", help="output folder (default reports/<timestamp>); must not exist yet")
    args = ap.parse_args()

    out = Path(args.out or Path("reports") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already has files; choose a new --out so earlier reports are kept")
    report = build_report(args.db, args.sessions, args.participants, args.title)
    if not report["study"]["sessions"]:
        sys.exit("no sessions match")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report, args.template), encoding="utf-8")
    print(f"wrote {out / 'report.md'} and {out / 'report.json'} "
          f"({report['study']['sessions']} sessions, {report['review']['spoken_turns']} spoken turns)")


if __name__ == "__main__":
    main()
