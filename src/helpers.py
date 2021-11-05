"""Helper functions for processing data."""
import numpy as np
import scipy


def serial_to_endcap(x: int) -> int:
    """Convert serialized chamber id to endcap."""
    return (x >> 10) + 1


def serial_to_station(x: int) -> int:
    """Convert serialized chamber id to station."""
    return ((x >> 8) & 0x00000003) + 1


def serial_to_ring(x: int) -> int:
    """Convert serialized chamber id to ring."""
    return ((x >> 6) & 0x00000003) + 1


def serial_to_chamber(x: int) -> int:
    """Convert serialized chamber id to chamber number."""
    return (x & 0x0000003F) + 1


def theta_to_eta(theta: float) -> float:
    """Convert theta to pseudorapidity eta."""
    return -np.log(np.tan(theta / 2.0))


def eta_to_theta(eta: float) -> float:
    """Convert pseudorapidity eta to theta."""
    return 2.0 * np.arctan(np.exp(-eta))


def pt_eta_to_p(pt: float, eta: float) -> float:
    """Convert transverse momentum and pseudorapidity to momentum."""
    return pt / np.sin(eta_to_theta(eta))


def landau(X, mean_offset, mean_slope, scale_slope, norm):
    """
    Landau pdf as a function of energy loss and muon momentum.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moyal.html#scipy.stats.moyal

    param x: x data to fit to
        [energy loss,
         energy]
    param mean_offset: offset from zero of the most probable value
    param mean_slope: linear dependency on momentum of most probable value
    param scale_slope: linear dependency on momentum of scale
    param norm: normalization

    return: probability of the energy loss de at a given e
    """
    dp, p = X
    return norm * scipy.stats.moyal.pdf(
        dp, loc=(mean_offset + mean_slope * p), scale=(scale_slope * p)
    )
