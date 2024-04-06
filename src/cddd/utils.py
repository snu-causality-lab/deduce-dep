import os
from typing import List

import pandas as pd
from filelock import FileLock


def safe_save_to_csv(result: List[List], columns: List[str], file_path: str):
    df = pd.DataFrame(result, columns=columns)
    lock = FileLock(file_path + '.lock')
    with lock:
        if not os.path.exists(file_path):
            df.to_csv(file_path, mode='w')
        else:
            df.to_csv(file_path, mode='a', header=False)

