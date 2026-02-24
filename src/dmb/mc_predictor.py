import jax
import jax.numpy as jnp
import jax.random as jrng

from astropy import units as u, constants as C
from tqdm.auto import tqdm


def _dedim(x: u.Quantity):
    return x.to(u.dimensionless_unscaled).value


def mc_prediction(
    NA,
    t_end: float,
    mA: u.Quantity["mass"] = 2 * C.m_p,
    mB: u.Quantity["mass"] = 1 * C.m_p,
    VA=jnp.array([200.0, 0.0, 0.0]),
    VB=jnp.array([0.0, 0.0, 0.0]),
    TA: float = 1e6,
    TB: float = 1e4,
    rhoA: float = 1e10,
    rhoB: float = 1e10,
    box: float = 10.0,
    sigma_0: float = 1e-26,
    sigma_n: int = 0,
    seed: int = 0,
):
    """
    Arguments:
        var      desc           unit          default
        NA       num samples    .             .
        t_end    end time       Gyr           .
        mA       A mass         g             2 m_p
        mB       B mass         g             m_p
        VA       A bulk vel     km/s          [200, 0, 0]
        VB       B bulk vel     km/s          [0, 0, 0]
        TA       A temperature  K             1e6
        TB       B temperature  K             1e4
        rhoA     A density      M_sun kpc^-3  1e10
        rhoB     B density      M_sun kpc^-3  1e10
        box      box size       kpc           10
        sigma_0  cross section  cm^2          1e-26
        sigma_n  vel dep power  .             0
        seed     random seed    .             0

    Returns: dict of form
        key       desc              unit
        times     times             Gyr
        means_A   A bulk vels       km/s
        means_B   B bulk vels       km/s
        disps_A   A vel disps       km^2/s^2
        disps_B   B vel disps       km^2/s^2
        temps_A   A temperatures    K
        temps_B   B temperatures    K
        energy_A  A total energy    M_sun km^2 s^-2
        energy_B  B total energy    M_sun km^2 s^-2
        P_A       A total momentum  M_sun km/s
        P_B       B total momentum  M_sun km/s
    """

    key = jrng.key(seed=seed)

    MA = rhoA * box**3
    MB = rhoB * box**3

    dispA = jnp.sqrt((C.k_B * TA * u.K / mA).to(u.km**2 / u.s**2).value)

    sigma_coeff = sigma_0 * u.cm**2 / C.c**sigma_n
    sigma_coeff = sigma_coeff.to(u.cm**2 / (u.km / u.s) ** sigma_n).value
    prob_coeff = _dedim(rhoB * u.M_sun / u.kpc**3 / mB * u.Gyr * u.cm**2 * u.km / u.s)

    @jax.vmap
    @jax.jit
    def cross_section(v):
        return sigma_coeff * jnp.power(v, sigma_n)

    def good_dt(key, vA, VB, TB):
        disp = jnp.sqrt(TB * (1 * u.erg / mB).to(u.km**2 / u.s**2).value)
        vB = disp * jrng.normal(key, (NA, 3)) + VB
        dv = vA - vB
        dv_mag = jnp.linalg.norm(dv, axis=-1)
        sigma = cross_section(dv_mag)
        P_ij = prob_coeff * sigma * dv_mag
        return 0.1 / P_ij.max()

    @jax.jit
    def timestep(key, vA, VB, TB, dt):
        """
        var  desc               dtype        unit
        vA   velocities         Array<NA,3>  km/s
        VB   gas bulk velocity  Array<3>     km/s
        TB   gas temperature    float        erg
        dt   timestep           float        Gyr
        """

        keys = jrng.split(key, 3)

        vB = (
            jnp.sqrt(TB * (1 * u.erg / mB).to(u.km**2 / u.s**2).value)
            * jrng.normal(keys[0], (NA, 3))
            + VB
        )
        dv = vA - vB
        dv_mag = jnp.linalg.norm(dv, axis=-1)
        sigma = cross_section(dv_mag)
        P_ij = prob_coeff * sigma * dv_mag * dt
        num_bad = jnp.count_nonzero(P_ij > 0.1)

        dv_prime = jrng.normal(keys[1], shape=(NA, 3))
        dv_prime = dv_prime * (dv_mag / jnp.linalg.norm(dv_prime, axis=-1))[:, None]

        kick = _dedim(mB / (mA + mB)) * (dv_prime - dv)
        r = jrng.uniform(keys[2], shape=(NA,))
        mask = (r < P_ij).astype(float)[:, None]
        vA_prime = vA + mask * kick

        # change in momentum
        VB_prime = VB - MA / MB / NA * (vA_prime - vA).sum(axis=0)

        # change in energy
        _MA = (mB * MA / MB * u.km**2 / u.s**2).to(u.erg).value
        _MB = (mB * MB / MB * u.km**2 / u.s**2).to(u.erg).value
        TB_prime = TB + (1 / 3) * (
            _MA / NA * (vA**2 - vA_prime**2).sum() + _MB * (VB**2 - VB_prime**2).sum()
        )

        # 0.5 * MA/NA * vA**2 + 0.5 * MB * VB**2 + 1.5 * MB/mB * TB
        # = 0.5 * MA/NA * vA_prime**2 + 0.5 * MB * VB_prime**2 + 1.5 * MB/mB * TB_prime
        # so TB_prime = TB + 1/3 * mB/MB * (MA/NA * (vA**2 - vA_prime**2) + MB * (VB**2 - VB_prime**2))
        return vA_prime, VB_prime, TB_prime, num_bad

    key, subA, subB = jrng.split(key, 3)

    _v = jrng.normal(subA, shape=(NA, 3))
    _v -= jnp.mean(_v, axis=0)
    _v /= jnp.std(_v)
    vA = dispA * _v + jnp.array(VA)

    VB = jnp.array(VB)
    TB = (C.k_B * TB * u.K).to(u.erg).value

    dt = good_dt(subB, vA, VB, TB) / 10
    n_timesteps = int(jnp.ceil(t_end / dt))

    data = {
        "times": [0.0],
        "means_A": [jnp.mean(vA, axis=0)],
        "disps_A": [jnp.var(vA, axis=0).sum() / 3],
        "means_B": [VB],
        "disps_B": [(TB * u.erg / mB).to(u.km**2 / u.s**2).value],
    }

    for _ in tqdm(range(n_timesteps)):
        key, sub = jrng.split(key)
        vA_prime, VB_prime, TB_prime, num_bad = timestep(sub, vA, VB, TB, dt)
        if num_bad > 10:
            print(f"{num_bad} bad probabilities")
            break

        meanA = jnp.mean(vA_prime, axis=0)
        dispA = jnp.var(vA_prime, axis=0).sum() / 3
        dispB = (TB_prime * u.erg / mB).to(u.km**2 / u.s**2).value

        vA = vA_prime
        VB = VB_prime
        TB = TB_prime

        data["times"].append(data["times"][-1] + dt)
        data["means_A"].append(meanA)
        data["means_B"].append(VB_prime)
        data["disps_A"].append(dispA)
        data["disps_B"].append(dispB)

    data = {k: jnp.stack(v) for k, v in data.items()}
    data["temps_A"] = (mA * data["disps_A"] * u.km**2 / u.s**2 / C.k_B).to(u.K)
    data["temps_B"] = (mB * data["disps_B"] * u.km**2 / u.s**2 / C.k_B).to(u.K)

    data["energy_A"] = (
        0.5 * MA * (jnp.square(data["means_A"]).sum(1) + 3 * data["disps_A"])
    )
    data["energy_B"] = (
        0.5 * MB * (jnp.square(data["means_B"]).sum(1) + 3 * data["disps_B"])
    )
    data["PA"] = MA * data["means_A"]
    data["PB"] = MB * data["means_B"]

    return data
