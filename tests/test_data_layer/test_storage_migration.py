import sqlite3

from src.data_layer.storage import StorageLayer


def test_model_prediction_observability_columns_are_migrated(tmp_path):
    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE model_prediction (
                   id INTEGER PRIMARY KEY,
                   code TEXT,
                   date TEXT,
                   pred_date TEXT,
                   pred_close REAL
            )"""
        )
        connection.execute(
            """CREATE TABLE price_prediction (
                   id INTEGER PRIMARY KEY,
                   code TEXT,
                   created_at TEXT,
                   pred_horizon INTEGER,
                   pred_price REAL
               )"""
        )
        StorageLayer._migrate_schema(connection)
    columns = {
        row[1] for row in connection.execute(
            "PRAGMA table_info(model_prediction)"
        )
    }

    assert {"last_close", "prediction_std", "actual_close", "actual_return", "score"} <= columns
    price_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(price_prediction)")
    }
    assert {
        "as_of_date", "pred_date", "pred_open", "pred_high", "pred_low",
        "pred_close", "prediction_std",
    } <= price_columns
