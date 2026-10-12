"""AutoML 2 prospective log: after the freeze, pre-register the frozen Mode A model's ranking for each upcoming draw.

This is the "next locked future period" (Step 12). The model is loaded from the frozen files and never refit;
calendar features need no past results, so every prediction stays strictly blind.

Run: python -m automl.model2.prospective
"""
from __future__ import annotations

import pandas as pd

from automl import common as c
from automl import live
from automl.model2.config import SEED
from automl.model2.trainer.models import Model

ART = c.ARTIFACTS / "automl2"
LOG = ART / "live" / "predictions.csv"


def main() -> None:
    model_dir = ART / "frozen" / "model"
    if not (model_dir / "model_meta.json").exists():
        print("AutoML 2 not frozen yet; nothing to pre-register")
        return
    manifest = c.read_json(ART / "manifest.json")
    s = c.load_stream()
    scored = live.score(LOG, s)
    date, draw_type = c.next_draw(s)
    rows = []
    if not live.already(live.load(LOG), date, draw_type, "champion"):
        model = Model.load(model_dir)
        p = model.predict(pd.DataFrame([{"date": date, "draw_type": draw_type}]))
        order = c.rank_order(p, seed=SEED)[0]
        rows.append(live.make_row(date, draw_type, "champion", f"frozen-{manifest['frozen_utc'][:10]}", order))
    live.append(LOG, rows)
    log = live.load(LOG)
    up = log[(log.draw_date == date) & (log.draw_type == draw_type)]
    c.write_json(ART / "live" / "latest.json", {
        "updated_utc": c.utcnow(), "next_draw": {"date": date, "draw_type": draw_type},
        "predictions": {r.kind: {"model_version": r.model_version, "created_utc": r.created_utc,
                                 "top100": r.top100.split(), "made_before_draw": r.made_before_draw}
                        for r in up.itertuples()},
        "scorecard": live.summary(LOG),
    })
    print(f"scored {scored}; pre-registered {len(rows)} for {date} {draw_type}")


if __name__ == "__main__":
    main()
