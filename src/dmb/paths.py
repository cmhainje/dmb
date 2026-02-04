from pathlib import Path

SRC_DMB  = Path(__file__).parent.absolute()
SRC      = SRC_DMB.parent.absolute()
ROOT     = SRC.parent.absolute()
AGORA    = (ROOT / "agora").absolute()
AGORA_IC = (AGORA / "ic.hdf5").absolute()
