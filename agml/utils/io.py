import os

def _is_valid_file(file):
    """Returns whether a file is valid.

    This means that it is a file and not a directory, but also that
    it isn't an unnecessary dummy file like `.DS_Store` on MacOS.
    """
    if os.path.isfile(file):
        if not file.startswith('.git') and file not in ['.DS_Store']:
            return True
    return False

def get_file_list(fpath):
    """Gets a list of files from a path."""
    return [f for f in os.listdir(fpath)
            if _is_valid_file(os.path.join(fpath, f))]

def get_dir_list(filepath):
    """Get a list of directories from a path."""
    return [f for f in os.listdir(filepath)
            if os.path.isdir(os.path.join(filepath, f))]

def nested_dir_list(fpath):
    """Returns a nested list of directories from a path."""
    dirs = []
    for f in os.scandir(fpath): # type: os.DirEntry
        if f.is_dir():
            dirs.append(os.path.join(fpath, f.path))
    if len(dirs) != 0:
        for dir_ in dirs:
            dirs.extend(nested_dir_list(dir_))
    return dirs

def nested_file_list(fpath):
    """Returns a nested list of files from a path."""
    files = []
    dirs = nested_dir_list(fpath)
    for dir_ in dirs:
        files.extend([i for i in os.listdir(dir_) if _is_valid_file(i)])

def create_dir(dir_):
    """Creates a directory (or does nothing if it exists)."""
    os.makedirs(dir_, exist_ok = True)
