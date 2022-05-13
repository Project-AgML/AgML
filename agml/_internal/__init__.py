


# Run the Helios Configuration.
def _check_helios_installation():
    """Checks for the latest Helios installation (and does as such).

    If no existing Helios installation is found, then this runs a fresh
    installation of Helios. If Helios is already installed, then this
    checks if a new possible version exists, and then installs if so.

    Note that this check if only run a maximum of once per day (specifically,
    the check is only run if it has not been run in the last 48 hours). This
    is to prevent constant resource-consuming checks for Git updates.
    """
    import os as _os
    import sys as _sys
    import json as _json
    import subprocess as _sp
    from datetime import datetime as _dt

    # Get the path to the Helios installation file.
    helios_file = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        '_helios/helios_install.sh')
    helios_dir = _os.path.join(
        _os.path.dirname(helios_file), 'Helios')

    # Run the installation/update. If the Helios directory is not found,
    # then run a clean installation. Otherwise, check if a new version of
    # Helios is available, then update the existing repository.
    if not _os.path.exists(helios_dir):
        _sys.stderr.write("Existing installation of Helios not found.\n")
        _sys.stderr.write("Installing Helios to:\n")
        _sys.stderr.write("\t" + helios_dir + "\n")

        # Update the last check of the Git update.
        with open(_os.path.expanduser('~/.agml/config.json')) as f:
            contents = _json.load(f)
        with open(_os.path.expanduser('~/.agml/config.json'), 'w') as f:
            contents['last_helios_check'] = _dt.now().strftime("%B %d %Y %H:%M:%S")
            _json.dump(contents, f, indent = 4)

    else:
        # Check if the Git update check has been run in the last 48 hours.
        with open(_os.path.expanduser('~/.agml/config.json')) as f:
            contents = _json.load(f)
        last_check = contents.get('last_helios_check', _dt(1970, 1, 1))
        last_check = _dt.strptime(last_check, "%B %d %Y %H:%M:%S")

        # If the last check has been run less than 48 hours ago, then
        # return without having updated Helios.
        if (_dt.now() - last_check).days == 0:
            return

        # Check if there is a new version available.
        _sys.stderr.write(
            f"Last check for Helios update: over {(_dt.now() - last_check).days} "
            f"day(s) ago. Checking for update.")

    # Execute the installation/update script.
    process = _sp.Popen(['bash', helios_file],
                        stdout = _sp.PIPE, universal_newlines = True)
    for line in iter(process.stdout.readline, ""):
        _sys.stderr.write(line)
    process.stdout.close()
    del _os, _sp


# Run the Helios check every time the module is imported.
_check_helios_installation()


# Import the synthetic data module.
from . import syntheticdata
