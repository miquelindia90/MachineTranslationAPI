


class DataIterator():
    def __init__(self, source_path: str, target_path: str, metadata_path: str, language_pair: str):
        self._source_path = source_path
        self._target_path = target_path
        self._metadata_path = metadata_path
        self._language_pair = language_pair