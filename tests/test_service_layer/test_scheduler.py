from src.scheduler.task_scheduler import TaskScheduler


def test_daily_schedule_uses_configured_timezone():
    scheduler = TaskScheduler()
    try:
        scheduler.start()
        job = scheduler.scheduler.get_job("a_share_daily_pipeline")

        assert job is not None
        assert str(job.trigger.timezone) == "Asia/Shanghai"
    finally:
        scheduler.shutdown()


def test_timing_data_update_refreshes_signal_and_execution_assets(monkeypatch):
    captured = {}

    class FakeDownloader:
        def __init__(self, config):
            pass

        def download(self, **kwargs):
            captured.update(kwargs)
            return {"failed_count": 0, "success_count": 2, "failed": []}

    monkeypatch.setattr("src.scheduler.task_scheduler.DataDownloader", FakeDownloader)
    scheduler = TaskScheduler()
    try:
        result = scheduler.update_timing_data()
    finally:
        scheduler.shutdown()

    assert captured["mode"] == "full"
    assert captured["symbols"] == ["000852", "512100"]
    assert result["success_count"] == 2
