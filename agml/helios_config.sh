#!/usr/bin/env bash

# Don't run this from any directory other than AgML, or else Helios
# will install in a different directory, which is not what we want.
if [ $# -eq 0 ]; then
  DIRNAME=$(basename "$PWD")
  if [ ! "$DIRNAME" = "agml" ]; then
    echo "Do not run this configuration script from anywhere but \`agml\`. Aborting"
    exit 1
  fi
  INSTALL_PATH=$PWD
else
  INSTALL_PATH=$1
fi

# Install Helios
echo "Installing Helios"
mkdir "$INSTALL_PATH"/_helios/Helios
git clone "https://github.com/PlantSimulationLab/Helios.git" "$INSTALL_PATH/_helios/Helios"