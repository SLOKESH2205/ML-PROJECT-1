import logging
import os
from datetime import datetime


def setup_logger():
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    LOG_FILE = os.path.join(
        LOG_DIR,
        f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )