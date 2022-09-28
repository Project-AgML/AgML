#!/usr/bin/env bash


# This script will almost certainly not be run manually, and instead
# be called by the AgML Helios module. So, we can't rely on using the
# existing parent directory to install Helios. Instead, we need to get
# the path of this actual file itself, and then
PATH_TO_ME="$(readlink -nf "$0")"
INSTALL_PATH="$(dirname "$(dirname "$PATH_TO_ME")")/_helios/Helios"

# Install or Update Helios, depending on whether the directory for Helios
# already exists. While the actual Python installation script which calls
# this shell script has slightly more complex logic (for figuring out
# version Helios is on, and in turn, whether it needs an update or not),
# this simply installs/updates based on the existence of the directory.
if [ ! -d "$INSTALL_PATH" ]; then
  git clone -b master https://github.com/PlantSimulationLab/Helios.git "$INSTALL_PATH"
else
  ORIGINAL_DIR="$PWD"
  cd "$INSTALL_PATH"
  git pull https://github.com/PlantSimulationLab/Helios.git master
  cd "$ORIGINAL_DIR" || echo "Issue when trying to update Helios. Please report this to the AgML team."; exit
fi



