"""End-to-end API smoke tests for FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from football_predictor.api.main import app


def test_api_endpoints_end_to_end() -> None:
    with TestClient(app) as client:
        # 1. Health check
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        health = resp_health.json()
        assert health["status"] == "ok"
        assert health["model_loaded"] is True
        assert health["n_silver_matches"] >= 2000

        # 2. Teams & power ratings
        resp_teams = client.get("/teams")
        assert resp_teams.status_code == 200
        teams_data = resp_teams.json()
        assert teams_data["n_teams"] >= 20

        # 3. Single match prediction with odds
        resp_pred = client.post(
            "/predict",
            json={
                "home_team": "Inter",
                "away_team": "Milan",
                "odds_home": 1.95,
                "odds_draw": 3.55,
                "odds_away": 4.10,
                "bankroll": 1000.0,
            },
        )
        assert resp_pred.status_code == 200
        pred = resp_pred.json()
        assert pred["home_team"] == "Inter"
        assert pred["away_team"] == "AC Milan"
        mk = pred["markets"]
        assert abs(mk["p_home"] + mk["p_draw"] + mk["p_away"] - 1.0) < 1e-5
        assert len(pred["score_matrix"]) == 9
        assert len(mk["top_scorelines"]) == 5
        assert len(pred["betting_signals"]) == 3

        # 4. Batch value-bets screener
        resp_vb = client.post(
            "/value-bets",
            json={
                "fixtures": [
                    {
                        "home_team": "Inter",
                        "away_team": "Hellas Verona",
                        "odds_home": 2.50,  # artificially generous odd to trigger value bet
                        "odds_draw": 3.80,
                        "odds_away": 4.50,
                    }
                ]
            },
        )
        assert resp_vb.status_code == 200
        vb = resp_vb.json()
        assert vb["n_fixtures"] == 1
        assert vb["n_value_bets"] >= 1

        # 5. Backtest summary & metrics
        resp_bt = client.get("/backtest/summary")
        assert resp_bt.status_code == 200
        bt = resp_bt.json()
        assert "calibration_metrics" in bt and "betting_metrics" in bt

        resp_metrics = client.get("/metrics")
        assert resp_metrics.status_code == 200
        assert "backtest" in resp_metrics.json()
