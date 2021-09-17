from . import data, train

def install_helios(overwrite = False):
    """Installs Helios into AgML."""
    # TODO, this is a temporary method until we configure the actual API.
    # Then this should be part of the general installation of AgML.
    import os as _os
    import shutil as _shutil
    if _os.path.exists(_os.path.join(
            _os.path.dirname(__file__), '_helios/Helios')):
        if not overwrite:
            raise FileExistsError(
                "Found existing installation of Helios, "
                "set `overwrite` to True to re-install.`")
        else:
            _shutil.rmtree(_os.path.join(
                _os.path.dirname(__file__), '_helios/Helios'))

    import subprocess as _subprocess
    _subprocess.call([
        f"{_os.path.join(_os.path.dirname(__file__), 'helios_config.sh')}",
        _os.path.dirname(__file__)])
    del _os, _shutil, _subprocess

install_helios()



