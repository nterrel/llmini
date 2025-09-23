# Test for logger setup in `utils.py`

def test_setup_logger():
    from llmini.utils import setup_logger
    logger = setup_logger("test_logger", level=10)
    assert logger.name == "test_logger"
    assert logger.level == 10
