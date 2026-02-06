import h5py
import numpy as np
import json

from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from shutil import rmtree
from string import Template

TEMPLATE_DIR = (Path(__file__).parent / "template").absolute()
TEMPLATES = sorted([Path(f).name for f in glob(str(TEMPLATE_DIR / "*"))])
TESTS_DIR = (Path(__file__).parent / "sims").absolute()
DOCKER_DIR = Path("/workspace/tests/sims")

BOX_SIZE = 10  # [kpc]
BOLTZMANN_CGS = 1.381e-16  # [erg/K]
PROTON_MASS_CGS = 1.672622e-24  # [g]
MEV_IN_G = 1.783e-27  # [g]
GEV_IN_G = 1.783e-24  # [g]


def make_grid(N: int, box_size: float):
    grid = np.linspace(0, box_size, 2 * N + 1)[1::2]
    X, Y, Z = np.meshgrid(grid, grid, grid)
    return np.stack(list(map(np.ndarray.flatten, [X, Y, Z])), axis=1)


def uniform_positions(rng: np.random.Generator, N: int, box_size: float):
    return rng.uniform(0.0, box_size, size=(N**3, 3))


def sample_velocities(rng: np.random.Generator, mean: float, T_over_m: float, N: int):
    if T_over_m <= 0:
        return mean * np.ones((N, 3))

    cov = np.eye(3) * T_over_m

    vel = None
    sample_temp = T_over_m * np.inf

    while (sample_temp - T_over_m) / T_over_m > 0.01:
        vel = rng.multivariate_normal(np.zeros(3), cov, size=(N,))
        vel -= np.mean(vel, 0)
        sample_temp = np.square(vel).sum(1).mean() / 3

    assert vel is not None

    vel += np.array([mean, 0, 0])
    return vel


class TestSim:
    def __init__(
        self,
        name: str,
        m_chi: float = 2 * PROTON_MASS_CGS,
        sigma_0: float = 10,
        vel_power: int = 0,
        vel_dm: float = 200,
        temp_dm: float = 1e6,
        temp_gas: float = 1e4,
        dens_dm: float = 1e10,
        dens_gas: float = 1e10,
        num_dm: int = 32,
        num_gas: int = 32,
        seed: int = 0,
    ):
        """
        m_chi:     [g]            mass of the dark matter particle
        sigma_0:   [mb]           total cross section at v=c
        vel_power: [.]            exponent of the cross section's dependence on (v/c)
        vel_dm:    [km/s]         mean velocity of the dark matter
        temp_dm:   [K]            temperature (vel disp) of the dark matter
        temp_gas:  [K]            temperature of the gas
        dens_dm:   [M_sun kpc^-3] density of the dark matter
        dens_gas:  [M_sun kpc^-3] density of the gas
        num_dm:    [.]            number of DM particles per axis (total number is this^3)
        num_gas:   [.]            number of gas particles per axis (total number is this^3)
        """

        self.name = name
        self.m_chi = m_chi
        self.sigma_0 = sigma_0
        self.vel_power = vel_power
        self.vel_dm = vel_dm
        self.temp_dm = temp_dm
        self.temp_gas = temp_gas
        self.dens_dm = dens_dm
        self.dens_gas = dens_gas
        self.num_dm = num_dm
        self.num_gas = num_gas
        self.host_path = str(TESTS_DIR / f"{name.replace(" ", "_")}")
        self.docker_path = str(DOCKER_DIR / f"{name.replace(" ", "_")}")

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def to_dict(self):
        return {
            "host_path": self.host_path,
            "path": self.docker_path,
            "m_chi": f"{self.m_chi:.6e}",
            "sigma_0": f"{self.sigma_0 * 1e-27:.6e}",
            "vel_power": f"{self.vel_power:d}",
            "hsml_gas": f"{BOX_SIZE / self.num_gas / 32:.6e}",
            "hsml_dm": f"{BOX_SIZE / self.num_dm / 32:.6e}",
            "seed": f"{self.seed:d}",
        }

    def exists(self):
        outdir = Path(self.host_path)
        if not outdir.exists():
            return False

        if not (outdir / "ic.hdf5").exists():
            return False

        for template in TEMPLATES:
            if not (outdir / template).exists():
                return False

        return True

    def make(self):
        # [kpc]
        pos_gas = make_grid(self.num_gas, BOX_SIZE)
        pos_dm = uniform_positions(self.rng, self.num_dm, BOX_SIZE)

        # [.]
        ids_gas = np.arange(len(pos_gas))
        ids_dm = np.arange(len(pos_dm)) + ids_gas.max()

        # [1e10 M_sun]
        mass_gas = self.dens_gas * BOX_SIZE**3 / len(pos_gas) / 1e10
        mass_dm = self.dens_dm * BOX_SIZE**3 / len(pos_dm) / 1e10

        # [km^2/s^2]
        disp_gas = BOLTZMANN_CGS * self.temp_gas / PROTON_MASS_CGS * 1e-10
        disp_dm = BOLTZMANN_CGS * self.temp_dm / self.m_chi * 1e-10

        # [km/s]
        vel_gas = sample_velocities(self.rng, 0, 0, len(pos_gas))
        vel_dm = sample_velocities(self.rng, self.vel_dm, disp_dm, len(pos_dm))

        # [km^2/s^2]
        integy_gas = (1.5 * BOLTZMANN_CGS * self.temp_gas / PROTON_MASS_CGS) * 1e-10

        outdir = Path(self.host_path)
        outdir.mkdir(parents=True, exist_ok=True)

        # make the initial conditions
        with h5py.File(outdir / "ic.hdf5", "w") as f:
            N_part = np.array([len(pos_gas), len(pos_dm), 0, 0, 0, 0])

            h = f.create_group("Header")
            h.attrs["NumPart_ThisFile"] = N_part
            h.attrs["NumPart_Total"] = N_part
            h.attrs["NumPart_Total_HighWord"] = np.zeros(6)
            h.attrs["MassTable"] = np.zeros(6)
            h.attrs["Time"] = 0.0
            h.attrs["NumFilesPerSnapshot"] = 1
            h.attrs["Flag_DoublePrecision"] = 0

            p0 = f.create_group("PartType0")
            p0.create_dataset("ParticleIDs", data=ids_gas)
            p0.create_dataset("Coordinates", data=pos_gas)
            p0.create_dataset("Velocities", data=vel_gas)
            p0.create_dataset("Masses", data=np.ones(len(pos_gas)) * mass_gas)
            p0.create_dataset("InternalEnergy", data=np.ones(len(pos_gas)) * integy_gas)

            p1 = f.create_group("PartType1")
            p1.create_dataset("ParticleIDs", data=ids_dm)
            p1.create_dataset("Coordinates", data=pos_dm)
            p1.create_dataset("Velocities", data=vel_dm)
            p1.create_dataset("Masses", data=np.ones(len(pos_dm)) * mass_dm)

        # render the templates
        for filename in TEMPLATES:
            with open(TEMPLATE_DIR / filename, "r") as f:
                r = Template(f.read()).substitute(**self.to_dict())

            outpath = outdir / filename
            with open(outpath, "w") as f:
                f.write(r)


test_sims = [
    TestSim("fiducial"),
    TestSim("fiducial_r1", seed=1),
    TestSim("fiducial_r2", seed=2),
    TestSim("fiducial_r3", seed=3),
    TestSim("fiducial_r4", seed=4),
    TestSim("fiducial_r5", seed=5),
    TestSim("fiducial_r6", seed=6),
    TestSim("fiducial_r7", seed=7),
    TestSim("fiducial_r8", seed=8),
    TestSim("fiducial_r9", seed=9),
    TestSim("thermal eq", vel_dm=0),
    TestSim("ram pressure", temp_dm=10, temp_gas=10),
    TestSim("low rho_chi", dens_dm=1e9),
    TestSim("low rho_b", dens_gas=1e9),
    TestSim("low N_chi", num_dm=24),
    TestSim("low res", num_dm=24, num_gas=24),
    TestSim(
        "heavy",
        m_chi=10 * GEV_IN_G,
        temp_dm=1e6 * (10 * GEV_IN_G) / (2 * PROTON_MASS_CGS),
    ),
    TestSim(
        "light",
        m_chi=100 * MEV_IN_G,
        temp_dm=1e6 * (100 * MEV_IN_G) / (2 * PROTON_MASS_CGS),
        sigma_0=2,
    ),
    TestSim(
        "lighter",
        m_chi=10 * MEV_IN_G,
        temp_dm=1e6 * (10 * MEV_IN_G) / (2 * PROTON_MASS_CGS),
        sigma_0=2,
    ),
    TestSim(
        "v_minus_2",
        sigma_0=2e-33 * 1e27,  # 2e-33 cm^2
        vel_power=-2,
    ),
    TestSim(
        "v_minus_4",
        sigma_0=5e-40 * 1e27,  # 5e-40 cm^2
        vel_power=-4,
    ),
]

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-o", "--output", default="./test_sims.json")
    ap.add_argument("-F", "--force", action="store_true")
    args = ap.parse_args()

    to_dump = []
    for sim in test_sims:
        if not sim.exists():
            print(f"making {sim.name} at {sim.host_path}")
            sim.make()
        elif args.force:
            print(f"sim found at {sim.host_path}. destroying...")
            rmtree(sim.host_path)
            print(f"making {sim.name} at {sim.host_path}")
            sim.make()
        to_dump.append(sim.to_dict())

    print(f"writing used parameters to {args.output}")
    with open(args.output, "w") as f:
        json.dump(to_dump, f)

    print("done")
