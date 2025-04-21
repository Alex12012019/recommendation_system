import pickle
from pathlib import Path
import time
import logging
from config import CACHE_DIR

logger = logging.getLogger(__name__)


def cache_step(cache_name):
    """Декоратор для кеширования результатов функций"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_path = Path(CACHE_DIR) / f'{cache_name}.pkl'
            if cache_path.exists():
                logger.info(f'Loading cached: {cache_name}')
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            logger.info(f'Computing: {cache_name}')
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            with open(cache_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Cached {cache_name} ({elapsed:.2f}s)')
            return result

        return wrapper

    return decorator