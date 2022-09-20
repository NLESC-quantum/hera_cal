from . import utils
from . import redcal

import numpy as np
from copy import deepcopy
import astropy.constants as const
from collections import defaultdict
from scipy.cluster.hierarchy import fclusterdata

SPEED_OF_LIGHT = const.c.si.value


def is_frequency_redundant(bl1, bl2, freqs, antpos, blvec_error_tol=1e-9):
    """
    Determine whether or not two baselines are frequency redundant. Checks that
    both baselines have the same heading, polarization, and have overlapping uv-modes

    Parameters:
    ----------
    bl1 : tuple
        Tuple of antenna indices and polarizations of the first baseline
    bl2 : tuple
        Tuple of antenna indices and polarizations of the second baseline
    freqs : np.ndarray
        Array of frequencies found in the data in units of Hz
    antpos : dict
        Antenna positions in the form {ant_index: np.array([x,y,z])}.
    blvec_error_tol : float, default=1e-9
        Largest allowable euclidean distance a unit baseline vector can be away from an existing
        cluster to be considered a unique orientation. See "fclusterdata" for more details.

    Returns:
        Boolean value determining whether or not the baselines are frequency
        redundant

    """
    # Split baselines in component antennas
    _ant1, _ant2, _pol = bl1
    ant1, ant2, pol = bl2

    # Get baseline vectors
    blvec1 = antpos[_ant2] - antpos[_ant1]
    blvec2 = antpos[ant2] - antpos[ant1]

    # Check umode overlap
    blmag1 = np.linalg.norm(blvec1)
    blmag2 = np.linalg.norm(blvec2)
    cond1 = (
        blmag1 * freqs.min() <= blmag2 * freqs.max()
        and blmag1 * freqs.max() >= blmag2 * freqs.max()
    )
    cond2 = (
        blmag1 * freqs.min() <= blmag2 * freqs.min()
        and blmag1 * freqs.max() >= blmag2 * freqs.min()
    )

    # Check polarization match
    cond3 = _pol == pol

    # Check headings
    norm_vec1 = blvec1 / np.linalg.norm(blvec1)
    norm_vec2 = blvec2 / np.linalg.norm(blvec2)
    clusters = fclusterdata(
        np.array([norm_vec1, norm_vec2]), blvec_error_tol, criterion="distance"
    )
    cond4 = clusters[0] == clusters[1]

    return (cond1 or cond2) and cond3 and cond4


def get_unique_orientations(
    antpos,
    reds=None,
    pols=["nn"],
    min_ubl_per_orient=1,
    blvec_error_tol=1e-9,
    bl_error_tol=1.0,
):
    """
    Sort baselines into groups with the same radial heading. These groups of baselines are potentially
    frequency redundant in a similar way to redcal.get_reds does. Returns a list of RadialRedundantGroup objects

    Parameters:
    ----------
    antpos : dict
        Antenna positions in the form {ant_index: np.array([x,y,z])}.
    reds : list of lists
        List of lists of spatially redundant baselines in the array. Can be found using redcal.get_reds
    pols : list, default=['nn']
        A list of polarizations e.g. ['nn', 'ne', 'en', 'ee']
    min_ubl_per_orient : int, default=1
        Minimum number of baselines per unique orientation
    blvec_error_tol : float, default=1e-9
        Largest allowable euclidean distance a unit baseline vector can be away from an existing
        cluster to be considered a unique orientation. See "fclusterdata" for more details.
    bl_error_tol: float, default=1.0
        The largest allowable difference between baselines in a redundant group
        (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

    """
    if reds is None:
        reds = redcal.get_reds(antpos, pols=pols, bl_error_tol=bl_error_tol)
    else:
        reds = deepcopy(reds)

    _uors = {}
    for pol in pols:
        ubl_pairs = [red[0] for red in reds if red[0][-1] == pol]

        # Compute normalized baseline vectors
        normalized_vecs = []
        for bls in ubl_pairs:
            ant1, ant2, pol = bls
            normalized_vecs.append(
                (antpos[ant2] - antpos[ant1])
                / np.linalg.norm(antpos[ant2] - antpos[ant1])
            )

        # Cluster orientations
        clusters = fclusterdata(normalized_vecs, blvec_error_tol, criterion="distance")
        uors = [[] for i in range(np.max(clusters))]

        for cluster, bl in zip(clusters, ubl_pairs):
            uors[cluster - 1].append(bl)

        uors = sorted(uors, key=len, reverse=True)

        # Find clusters with headings anti-parallel to others
        for group in uors:
            ant1, ant2, pol = group[0]
            vec = (antpos[ant2] - antpos[ant1]) / np.linalg.norm(
                antpos[ant2] - antpos[ant1]
            )
            vec = np.array(vec / blvec_error_tol, dtype=int)
            if tuple(-vec) + (pol,) in _uors:
                _uors[tuple(-vec) + (pol,)] += [utils.reverse_bl(bls) for bls in group]
            else:
                _uors[tuple(vec) + (pol,)] = group

    # Convert lists to RadialRedundantGroup objects
    uors = [
        RadialRedundantGroup(_uors[key], antpos)
        for key in _uors
        if len(_uors[key]) >= min_ubl_per_orient
    ]
    uors = sorted(uors, key=len, reverse=True)
    return uors


class RadialRedundantGroup:
    """ """

    def __init__(self, baselines, antpos, blvec=None, pol=None):
        """ """
        _baselines = deepcopy(baselines)

        # Attach polarization and normalized vector to radial redundant group
        if pol is None:
            pols = list(set([bl[2] for bl in baselines]))
            if len(pols) > 1:
                raise ValueError(
                    f"Multiple polarizations are in your radially redundant group: {pols}"
                )
            else:
                self.pol = pols[0]
        else:
            self.pol = pol

        if blvec is None:
            ant1, ant2, pol = baselines[0]
            self.blvec = (antpos[ant2] - antpos[ant1]) / np.linalg.norm(
                antpos[ant2] - antpos[ant1]
            )
        else:
            self.blvec = blvec

        # Store baseline lengths
        baseline_lengths = []
        for baseline in baselines:
            ant1, ant2, pol = baseline
            baseline_lengths.append(np.linalg.norm(antpos[ant2] - antpos[ant1]))

        # Sort baselines list by baseline length
        self._baselines = [_baselines[idx] for idx in np.argsort(baseline_lengths)]
        self.baseline_lengths = [
            baseline_lengths[idx] for idx in np.argsort(baseline_lengths)
        ]

    def get_u_bounds(self, freqs):
        """
        Calculates the magnitude of the minimum and maximum u-modes values of the radial redundant group
        given an array of frequency values

        Parameters:
        ----------
            freqs: np.ndarray
                Array of frequencies found in the data in units of Hz

        Returns:
            ubounds: tuple
                Tuple of the magnitude minimum and maximum u-modes sampled by this baseline group
        """
        umin = freqs.min() / 2.998e8 * np.min(self.baseline_lengths)
        umax = freqs.max() / 2.998e8 * np.max(self.baseline_lengths)
        return (umin, umax)

    def filter_group(
        self,
        bls=None,
        ex_bls=None,
        ants=None,
        ex_ants=None,
        ubls=None,
        ex_ubls=None,
        pols=None,
        ex_pols=None,
        antpos=None,
        min_bl_cut=None,
        max_bl_cut=None,
    ):
        """ """
        _baselines = redcal.filter_reds(
            [self._baselines],
            bls=bls,
            ex_bls=ex_bls,
            ants=ants,
            ex_ants=ex_ants,
            ubls=ubls,
            ex_ubls=ex_ubls,
            pols=pols,
            ex_pols=ex_pols,
        )
        if len(_baselines) == 0:
            self._baselines = []
            self.baseline_lengths = []
        else:
            new_bls = []
            new_bls_lengths = []
            for bls in _baselines[0]:
                index = self._baselines.index(bls)
                if min_bl_cut is not None and self.baseline_lengths[index] < min_bl_cut:
                    continue
                if max_bl_cut is not None and self.baseline_lengths[index] > max_bl_cut:
                    continue
                new_bls.append(bls)
                new_bls_lengths.append(self.baseline_lengths[index])

            self._baselines = new_bls
            self.baseline_lengths = new_bls_lengths

    def __iter__(self):
        """Iterate through baselines in the radially redundant group"""
        return iter(self._baselines)

    def __len__(self):
        """Return the length of the baselines list"""
        return len(self._baselines)

    def __getitem__(self, index):
        """Get the baseline at the chosen index"""
        return self._baselines[index]


class FrequencyRedundancy:
    """ """

    def __init__(
        self, antpos, reds=None, blvec_error_tol=1e-9, pols=["nn"], bl_error_tol=1.0
    ):
        """
        Parameters:
        ----------
        antpos : dict
            Antenna positions in the form {ant_index: np.array([x,y,z])}.
        reds : list of list
            List of lists of baseline keys. Can be determined using redcal.get_reds
        pols : list of strs
            List of polarization strings to be used in the frequency redundant group
        """
        self.antpos = antpos

        if reds is None:
            reds = redcal.get_reds(antpos, pols=pols, bl_error_tol=bl_error_tol)

        self._mapped_reds = {red[0]: red for red in reds}

        self._radial_groups = get_unique_orientations(
            antpos, reds=reds, pols=pols, blvec_error_tol=blvec_error_tol
        )

    def get_radial_group(self, key):
        """
        Get baselines with the same heading as a given baseline

        Parameters:
            key: tuple
                Basleine key of type (ant1, ant2, pol)
        """
        # Identify headings
        for group_key in self._mapped_reds:
            if key == group_key or key in self._mapped_reds[group_key]:
                key = group_key
                break

        for group in self._radial_groups:
            if key in group:
                return group

    def get_redundant_group(self, key):
        """
        Get a list of baseline that are spatially redundant with the input baseline

        Parameters:
            key: tuple
                Baseline key with of type (ant1, ant2, pol)
        """
        for group_key in self._mapped_reds:
            if key == group_key or key in self._mapped_reds[group_key]:
                key = group_key
                break

        if utils.reverse_bl(key) in self._mapped_reds:
            return [
                utils.reverse_bl(bls)
                for bls in self._mapped_reds.get(utils.reverse_bl(key))
            ]
        elif key in self._mapped_reds:
            return self._mapped_reds.get(key)
        else:
            raise KeyError(
                f"Baseline {key} is not in the group of spatial redundancies"
            )

    def get_pol(self, pol):
        """Get all radially redundant groups with a given polarization"""
        for group in self:
            if group.pol == pol:
                yield group

    def filter_radial_groups(
        self,
        min_nbls=4,
        bls=None,
        ex_bls=None,
        ants=None,
        ex_ants=None,
        ubls=None,
        ex_ubls=None,
        pols=None,
        ex_pols=None,
        antpos=None,
        min_bl_cut=None,
        max_bl_cut=None,
    ):
        """
        Find all radial groups with

        Parameters:
        ----------
            min_nbls: pass
                pass
            ex_bls: pass
                pass
            ex_ants: pass
                pass
        """
        _bad_groups = []
        for gi, group in enumerate(self._radial_groups):
            # Filter radially redundant group
            group.filter_group(
                bls=bls,
                ex_bls=ex_bls,
                ants=ants,
                ex_ants=ex_ants,
                ubls=ubls,
                ex_ubls=ex_ubls,
                pols=pols,
                ex_pols=ex_pols,
                min_bl_cut=min_bl_cut,
                max_bl_cut=max_bl_cut,
            )
            # Identify groups with fewer than min_nbls baselines
            if len(group) < min_nbls:
                _bad_groups.append(gi)
            else:
                pass

        # Remove antennas with fewer than min_nbls
        for _bad_group in sorted(_bad_groups, reverse=True):
            self._radial_groups.pop(_bad_group)

    def __iter__(self):
        """Iterates through the list of redundant groups"""
        return iter(self._radial_groups)
