import logging
logger = logging.getLogger("detectron2")

logging.basicConfig(
    level=logging.INFO,
    filename="test.log",
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info("test")
