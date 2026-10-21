from src import config


def test_paths_are_inside_project_root():
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.RESULTS_DIR):
        assert path.is_relative_to(config.ROOT)


def test_project_folders_exist():
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.RESULTS_DIR):
        assert path.is_dir(), f"missing folder: {path}"


def test_timezone_is_karachi():
    assert config.TIMEZONE == "Asia/Karachi"
