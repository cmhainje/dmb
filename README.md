# dmb

A new method for incorporating dark matter–baryon scattering in hydrodynamic simulations.
The method is implemented in a fork of [GIZMO](https://github.com/pfhopkins/gizmo-public) here: [cmhainje/gizmo-public](https://github.com/cmhainje/gizmo-public).

This repository contains

- A Docker+VS Code-based environment for GIZMO development
- Scripts for creating and analyzing the implementation in a highly-idealized system
- Scripts for running and analyzing simulations of an isolated disk galaxy


## Authors

- **Connor Hainje**, NYU
- **Glennys Farrar**, NYU

## Usage

Clone this repo and the GIZMO fork (should happen automatically).

If you open this project in VS Code, it should automatically detect the devcontainer setup and offer to re-launch the workspace inside a Docker container.
The Docker container has the necessary dependencies to compile and run GIZMO; I have found this very useful for testing.
You only need to provide a Config.sh file and run `make`, and all the rest should work! 

## License

This work is distributed under the MIT license.
However, our fork of GIZMO inherits the GNU General Public License, as detailed in the [README therein](https://github.com/cmhainje/gizmo-public/blob/dmb/README.md).
