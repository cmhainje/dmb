# dmb

A new method for incorporating dark matter–baryon scattering in hydrodynamic simulations.
The method is implemented in a fork of [GIZMO](https://github.com/pfhopkins/gizmo-public) here: [cmhainje/gizmo-public](https://github.com/cmhainje/gizmo-public).

This repository contains

- A Docker container for managing builds which works on ARM (Mac) and x86 (Linux) architectures and can be used with VS Code's Dev Containers extension.
- Scripts for making, running, and analyzing test problems
- Scripts for running and analyzing simulations of an isolated disk galaxy

## Authors

- **Connor Hainje**, NYU
- **Glennys Farrar**, NYU

## Usage

Clone this repo:

```bash
git clone --recursive https://github.com/cmhainje/dmb.git
```

### VS Code

If you open this project in VS Code, it should automatically detect the Dev Container setup and offer to re-launch the workspace inside a Docker container.
The Docker container has the necessary dependencies to compile and run GIZMO; I have found this very useful for testing.
You only need to provide a Config.sh file and run `make`, and all the rest should work!

### Using Docker

You can also use the Docker container manually for managing dependencies.
I have it pushed to Docker Hub, so you can pull it from there:

```bash
docker pull cmhainje/cenv:latest
```

and then use it

```bash
# compile
cd path/to/dmb/gizmo-public
docker run -v $(pwd):/workspace cmhainje/cenv:latest bash -c 'make clean && make'

# run
cd path/to/dmb/gizmo-public
docker run -v $(pwd):/workspace cmhainje/cenv:latest ./GIZMO
```

If you want to build it yourself,

```bash
docker build --platform linux/arm64,linux/amd64 -t <tagname>:latest .
```

Just be sure to update the tagname in the previous commands if you decide to change it in the build.

### NYU Torch

I'm working on NYU's Torch cluster, which uses Apptainer for container-related things.
The system provides an image which contains the relevant dependencies.
The dependencies are located at the same paths as in my Docker image, so the Makefiles still work.
My cheat sheet:

```bash
# compile
cd /path/to/dmb/gizmo-public
apptainer exec \
  /share/apps/images/ubuntu-24.04.3.sif \
  make

# run
apptainer exec \
  /share/apps/images/ubuntu-24.04.3.sif \
  /path/to/GIZMO \
  /path/to/params/gizmo.param

# or using OpenMPI for multiprocessing
apptainer exec \
  /share/apps/images/ubuntu-24.04.3.sif \
  mpiexec -bind-to none -np 16 \
  /path/to/GIZMO \
  /path/to/params/gizmo.param
```

## License

This work is distributed under the MIT license.
However, our fork of GIZMO inherits the GNU General Public License, as detailed in the [README therein](https://github.com/cmhainje/gizmo-public/blob/dmb/README.md).
