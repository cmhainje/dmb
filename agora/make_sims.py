import json

from argparse import ArgumentParser
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from shutil import rmtree
from string import Template

TEMPLATE_DIR = (Path(__file__).parent / "template").absolute()
DEFAULT_GIZMO = "/home/ch4407/local/bin/GIZMO"
SIMS_DIR = Path("/scratch/ch4407/dmb").absolute()

TEMPLATES = sorted([Path(f).name for f in glob(str(TEMPLATE_DIR / "*"))])


class Sim:
    def __init__(
        self,
        sigma_mb: float,
        name: str | None = None,
        directory: str | None = None,
        exepath: str = DEFAULT_GIZMO,
        random_seed: int = 0,
    ):
        self.sigma_mb = sigma_mb
        self.exepath = exepath
        self.random_seed = random_seed

        fallback_name = f"agora_{sigma_mb:.0f}"
        if random_seed != 0:
            fallback_name += f"_{random_seed}"
        self.name = name or fallback_name
        self.directory = directory or str(SIMS_DIR / self.name)

        self.sigma_cgs = sigma_mb * 1e-27

    def to_dict(self):
        return {
            "sigma": f"{self.sigma_cgs:.0e}",
            "name": self.name,
            "directory": self.directory,
            "exepath": self.exepath,
            "random_seed": self.random_seed,
        }

    def is_rendered(self):
        """check that the directory exists and all templates exist within it"""
        outdir = Path(self.directory)
        if not outdir.exists():
            return False
        for filename in TEMPLATES:
            if not (outdir / "filename").exists():
                return False
        return True

    def render(self, dry_run=False, verbose=False):
        outdir = Path(self.directory)
        if not dry_run:
            outdir.mkdir(parents=True, exist_ok=True)

        for filename in TEMPLATES:
            with open(TEMPLATE_DIR / filename, "r") as f:
                r = Template(f.read()).substitute(**self.to_dict())

            outpath = outdir / filename
            if dry_run:
                print(f"would write {outpath}")
                if verbose:
                    print(r)
            else:
                if verbose:
                    print(f"writing {outpath}")
                with open(outpath, "w") as f:
                    f.write(r)


sims = [
    # main suite
    Sim(0),
    Sim(10),
    Sim(30),
    Sim(50),
    Sim(70),
    Sim(100),
    Sim(300),
]


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-o", "--output", default="./made_sims.json")
    ap.add_argument("-F", "--force", action="store_true")
    ap.add_argument("-n", "--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    to_dump = []
    for sim in sims:
        if args.dry_run:
            sim.render(dry_run=True, verbose=args.verbose)
            continue

        if sim.is_rendered():
            if args.force:
                rmtree(sim.directory)
            else:
                to_dump.append(sim.to_dict())
                continue

        sim.render(verbose=args.verbose)
        to_dump.append(sim.to_dict())

    if not args.dry_run:
        if args.verbose:
            print(f"dumping parameters to {args.output}")
        with open(args.output, "w") as f:
            json.dump(to_dump, f)
