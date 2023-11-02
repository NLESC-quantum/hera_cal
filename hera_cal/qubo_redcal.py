# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
import linsolve.quantum

from . import utils
from .datacontainer import DataContainer
from .utils import split_bl

from .redcal import *
from .redcal import _firstcal_align_bls, _find_flipped, _wrap_phs


class QUBOOmnicalSolver(linsolve.quantum.QUBOLinProductSolver, OmnicalSolver):
    def __init__(self, solver, data, sol0, wgts={}, gain=0.3, **kwargs):
        """Set up a nonlinear system of equations of the form g_i * g_j.conj() * V_mdl = V_ij
        to linearize via the Omnical algorithm described in HERA Memo 50
        (scripts/notebook/omnical_convergence.ipynb).

        Args:
            data: Dictionary that maps nonlinear product equations, written as valid python-interpetable
                strings that include the variables in question, to (complex) numbers or numpy arrarys.
                Variables with trailing underscores '_' are interpreted as complex conjugates (e.g. x*y_
                parses as x * y.conj()).
            sol0: Dictionary mapping all variables (as keyword strings) to their starting guess values.
                This is the point that is Taylor expanded around, so it must be relatively close to the
                true chi^2 minimizing solution. In the same format as that produced by
                linsolve.LogProductSolver.solve() or linsolve.LinProductSolver.solve().
            wgts: Dictionary that maps equation strings from data to real weights to apply to each
                equation. Weights are treated as 1/sigma^2. All equations in the data must have a weight
                if wgts is not the default, {}, which means all 1.0s.
            gain: The fractional step made toward the new solution each iteration.  Default is 0.3.
                Values in the range 0.1 to 0.5 are generally safe.  Increasing values trade speed
                for stability.
            **kwargs: keyword arguments of constants (python variables in keys of data that
                are not to be solved for) which are passed to linsolve.LinProductSolver.
        """

        linsolve.quantum.QuantumLinProductSolver.__init__(
            self, solver, data, sol0, wgts=wgts, **kwargs
        )

        self.gain = np.float32(
            gain
        )  # float32 to avoid accidentally promoting data to doubles.

    def solve_iteratively(
        self,
        conv_crit=1e-10,
        maxiter=50,
        check_every=4,
        check_after=1,
        wgt_func=lambda x: 1.0,
        verbose=False,
    ):
        """Repeatedly solves and updates solution until convergence or maxiter is reached.
        Returns a meta-data about the solution and the solution itself.

        Args:
            conv_crit: A convergence criterion (default 1e-10) below which to stop iterating.
                Converegence is measured L2-norm of the change in the solution of all the variables
                divided by the L2-norm of the solution itself.
            maxiter: An integer maximum number of iterations to perform before quitting. Default 50.
            check_every: Compute convergence and updates weights every Nth iteration (saves computation). Default 4.
            check_after: Start computing convergence and updating weights after the first N iterations.  Default 1.
            wgt_func: a function f(abs^2 * wgt) operating on weighted absolute differences between
                data and model that returns an additional data weighting to apply to when calculating
                chisq and updating parameters. Example: lambda x: np.where(x>0, 5*np.tanh(x/5)/x, 1)
                clamps deviations to 5 sigma. Default is no additional weighting (lambda x: 1.).

        Returns: meta, sol
            meta: a dictionary with metadata about the solution, including
                iter: the number of iterations taken to reach convergence (or maxiter), with dimensions of the data.
                chisq: the chi^2 of the solution produced by the final iteration, with dimensions of the data.
                conv_crit: the convergence criterion evaluated at the final iteration, with dimensions of the data.
            sol: a dictionary of complex solutions with variables as keys, with dimensions of the data.
        """
        sol = self.sol0
        terms = [
            (linsolve.get_name(gi), linsolve.get_name(gj), linsolve.get_name(uij))
            for term in self.all_terms
            for (gi, gj, uij) in term
        ]
        dmdl_u = self._get_ans0(sol)
        abs2_u = {
            k: np.abs(self.data[k] - dmdl_u[k]) ** 2 * self.wgts[k] for k in self.keys
        }
        chisq = sum([v * wgt_func(v) for v in abs2_u.values()])
        update = np.where(chisq > 0)
        abs2_u = {k: v[update] for k, v in abs2_u.items()}
        # variables with '_u' are flattened and only include pixels that need updating
        dmdl_u = {k: v[update].flatten() for k, v in dmdl_u.items()}
        # wgts_u hold the wgts the user provides
        wgts_u = {
            k: (v * np.ones(chisq.shape, dtype=np.float32))[update].flatten()
            for k, v in self.wgts.items()
        }
        # clamp_wgts_u adds additional sigma clamping done by wgt_func.
        # abs2_u holds abs(data - mdl)**2 * wgt (i.e. noise-weighted deviations), which is
        # passed to wgt_func to determine any additional weighting (to, e.g., clamp outliers).
        clamp_wgts_u = {k: v * wgt_func(abs2_u[k]) for k, v in wgts_u.items()}
        sol_u = {k: v[update].flatten() for k, v in sol.items()}
        iters = np.zeros(chisq.shape, dtype=int)
        conv = np.ones_like(chisq)
        for i in range(1, maxiter + 1):
            if verbose:
                print("Beginning iteration %d/%d" % (i, maxiter))
            if (i % check_every) == 1:
                # compute data wgts: dwgts = sum(V_mdl^2 / n^2) = sum(V_mdl^2 * wgts)
                # don't need to update data weighting with every iteration
                # clamped weighting is passed to dwgts_u, which is used to update parameters
                dwgts_u = {
                    k: dmdl_u[k] * dmdl_u[k].conj() * clamp_wgts_u[k] for k in self.keys
                }
                sol_wgt_u = {k: 0 for k in sol.keys()}
                for k, (gi, gj, uij) in zip(self.keys, terms):
                    w = dwgts_u[k]
                    sol_wgt_u[gi] += w
                    sol_wgt_u[gj] += w
                    sol_wgt_u[uij] += w
                dw_u = {k: v[update] * dwgts_u[k] for k, v in self.data.items()}
            sol_sum_u = {k: 0 for k in sol_u.keys()}
            for k, (gi, gj, uij) in zip(self.keys, terms):
                # compute sum(wgts * V_meas / V_mdl)
                numerator = dw_u[k] / dmdl_u[k]
                sol_sum_u[gi] += numerator
                sol_sum_u[gj] += numerator.conj()
                sol_sum_u[uij] += numerator
            new_sol_u = {
                k: v * ((1 - self.gain) + self.gain * sol_sum_u[k] / sol_wgt_u[k])
                for k, v in sol_u.items()
            }
            dmdl_u = self._get_ans0(new_sol_u)
            # check if i % check_every is 0, which is purposely one less than the '1' up at the top of the loop
            if i < maxiter and (i < check_after or (i % check_every) != 0):
                # Fast branch when we aren't expensively computing convergence/chisq
                sol_u = new_sol_u
            else:
                # Slow branch when we compute convergence/chisq
                abs2_u = {
                    k: np.abs(v[update] - dmdl_u[k]) ** 2 * wgts_u[k]
                    for k, v in self.data.items()
                }
                new_chisq_u = sum([v * wgt_func(v) for v in abs2_u.values()])
                chisq_u = chisq[update]
                gotbetter_u = chisq_u > new_chisq_u
                where_gotbetter_u = np.where(gotbetter_u)
                update_where = tuple(u[where_gotbetter_u] for u in update)
                chisq[update_where] = new_chisq_u[where_gotbetter_u]
                iters[update_where] = i
                new_sol_u = {
                    k: np.where(gotbetter_u, v, sol_u[k]) for k, v in new_sol_u.items()
                }
                deltas_u = [v - sol_u[k] for k, v in new_sol_u.items()]
                conv_u = np.sqrt(
                    sum([(v * v.conj()).real for v in deltas_u])
                    / sum([(v * v.conj()).real for v in new_sol_u.values()])
                )
                conv[update_where] = conv_u[where_gotbetter_u]
                for k, v in new_sol_u.items():
                    sol[k][update] = v
                update_u = np.where((conv_u > conv_crit) & gotbetter_u)
                if update_u[0].size == 0 or i == maxiter:
                    meta = {"iter": iters, "chisq": chisq, "conv_crit": conv}
                    return meta, sol
                dmdl_u = {k: v[update_u] for k, v in dmdl_u.items()}
                wgts_u = {k: v[update_u] for k, v in wgts_u.items()}
                sol_u = {k: v[update_u] for k, v in new_sol_u.items()}
                abs2_u = {k: v[update_u] for k, v in abs2_u.items()}
                clamp_wgts_u = {k: v * wgt_func(abs2_u[k]) for k, v in wgts_u.items()}
                update = tuple(u[update_u] for u in update)
            if verbose:
                print(
                    "    <CHISQ> = %f, <CONV> = %f, CNT = %d",
                    (np.mean(chisq), np.mean(conv), update[0].size),
                )


class QUBORedundantCalibrator(RedundantCalibrator):
    def __init__(self, reds, solver=None, check_redundancy=False, **kwargs):
        """Initialization of a class object for performing redundant calibration with logcal
        and lincal, both utilizing linsolve, and also degeneracy removal.

        Args:
            solver: a vqls instance
            reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol). The first
                item in each list will be treated as the key for the unique baseline
            check_redundancy: if True, raise an error if the array is not redundantly calibratable,
                even when allowing for an arbitrary number of phase slope degeneracies.
        """
        super().__init__(reds, check_redundancy)
        self.solver = solver

    def _solver(self, linsolve_method, data, wgts={}, detrend_phs=False, **kwargs):
        """Instantiates a linsolve solver for performing redcal.

        Args:
            linsolve method: one of the linsolve solver (linsolve.quantum.XXX or QuantumOmnicalSolver)
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            detrend_phs: takes out average phase, useful for logcal
            **kwargs: other keyword arguments passed into the solver for use by linsolve, e.g.
                sparse (use sparse matrices to represent system of equations).

        Returns:
            solver: instantiated solver with redcal equations and weights
        """
        dtype = list(data.values())[0].dtype
        dc = DataContainer(data)
        eqs = self.build_eqs(dc)
        self.phs_avg = (
            {}
        )  # detrend phases within redundant group, used for logcal to avoid phase wraps
        if detrend_phs:
            for grp in self.reds:
                self.phs_avg[grp[0]] = np.exp(
                    -np.complex64(1j)
                    * np.median(
                        np.unwrap([np.log(dc[bl]).imag for bl in grp], axis=0), axis=0
                    )
                )
                for bl in grp:
                    self.phs_avg[bl] = self.phs_avg[grp[0]].astype(dc[bl].dtype)
        d_ls, w_ls = {}, {}
        for eq, key in eqs.items():
            d_ls[eq] = dc[key] * self.phs_avg.get(key, np.float32(1))
        if len(wgts) > 0:
            wc = DataContainer(wgts)
            for eq, key in eqs.items():
                w_ls[eq] = wc[key]
        return linsolve_method(self.solver, data=d_ls, wgts=w_ls, **kwargs)



    def firstcal(
        self,
        data,
        freqs,
        maxiter=100,
        sparse=False,
        mode="qubo",
        flip_pnt=(np.pi / 2),
        return_matrix = False,
    ):
        """Solve for a calibration solution parameterized by a single delay and phase offset
        per antenna using the phase difference between nominally redundant measurements.
        Delays are solved in a single iteration, but phase offsets are solved for
        iteratively to account for phase wraps.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            freqs: numpy array of frequencies in the data
            maxiter: maximum number of iterations for finding flipped antennas
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            mode: solving mode passed to the quantum linsolve linear solver ('vqls')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().
            flip_pnt: cutoff median phase to assign baselines the "majority" polarity group.
                (pi - max_rel_angle() is the cutoff for "minority" group. Must be between 0 and pi/2.

        Returns:
            meta: dictionary of metadata (including delays and suspected antenna flips for each integration)
            sol: RedSol with Ntimes x Nfreqs per-antenna gains solutions of the form
                 np.exp(2j * np.pi * delay * freqs + 1j * offset), as well as visibility
                 solutions formed from redundantly averaged first-caled data.
        """
        Ntimes, Nfreqs = data[self.reds[0][0]].shape
        dlys_offs = {}

        for bls in self.reds:
            if len(bls) < 2:
                continue
            _dly_off = _firstcal_align_bls(bls, freqs, data)
            dlys_offs.update(_dly_off)

        # offsets often have phase wraps and need some finesse around np.pi
        avg_offsets = {
            k: np.mean(v[1]) for k, v in dlys_offs.items()
        }  # XXX maybe do per-integration
        flipped = _find_flipped(avg_offsets, flip_pnt=flip_pnt, maxiter=maxiter)

        d_ls = {}
        for (bl1, bl2), (dly, off) in dlys_offs.items():
            ai, aj = split_bl(bl1)
            am, an = split_bl(bl2)
            i, j, m, n = (self.pack_sol_key(k) for k in (ai, aj, am, an))
            eq_key = "%s-%s-%s+%s" % (i, j, m, n)
            n_flipped = sum([int(ant in flipped) for ant in (ai, aj, am, an)])
            if n_flipped % 2 == 0:
                d_ls[eq_key] = np.array((dly, off))
            else:
                d_ls[eq_key] = np.array((dly, _wrap_phs(off + np.pi)))
        ls = linsolve.quantum.QUBOLinearSolver(self.solver, d_ls, sparse=sparse)
        
        if return_matrix:
            return ls.return_matrix()
        
        sol, res = ls.solve(mode=mode)
        dlys = {self.unpack_sol_key(k): v[0] for k, v in sol.items()}
        offs = {self.unpack_sol_key(k): v[1] for k, v in sol.items()}
        # add back in antennas in reds but not in the system of equations
        ants = set(
            [ant for red in self.reds for bl in red for ant in utils.split_bl(bl)]
        )
        dlys = {
            ant: dlys.get(ant, (np.zeros_like(list(dlys.values())[0]))) for ant in ants
        }
        offs = {
            ant: offs.get(ant, (np.zeros_like(list(offs.values())[0]))) for ant in ants
        }

        for ant in flipped:
            offs[ant] = _wrap_phs(offs[ant] + np.pi)

        dtype = np.find_common_type([d.dtype for d in data.values()], [])
        meta = {
            "dlys": {ant: dly.flatten() for ant, dly in dlys.items()},
            "offs": {ant: off.flatten() for ant, off in offs.items()},
            "polarity_flips": {
                ant: np.ones(Ntimes, dtype=bool) * bool(ant in flipped) for ant in ants
            },
        }
        gains = {
            ant: np.exp(2j * np.pi * dly * freqs + 1j * offs[ant]).astype(dtype)
            for ant, dly in dlys.items()
        }
        sol = RedSol(self.reds, gains=gains)
        sol.update_vis_from_data(
            data
        )  # compute vis sols for completeness (though not strictly necessary)
        return meta, sol, res

    def logcal(self, data, sol0=None, wgts={}, sparse=False, mode="qubo", return_matrix=False):
        """Takes the log to linearize redcal equations and minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary that includes all starting (e.g. firstcal) gains in the
                {(ant,antpol): np.array} format. These are divided out of the data before
                logcal and then multiplied back into the returned gains in the solution.
                Missing gains are treated as 1.0s.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            mode: solving mode passed to the linsolve linear solver ('default', 'lsqr', 'pinv', or 'solve')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().

        Returns:
            meta: empty dictionary (to maintain consistency with related functions)
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        cal_data = {bl: data[bl] for gp in self.reds for bl in gp}
        if sol0 is not None:
            cal_data = {bl: sol0.calibrate_bl(bl, data[bl]) for bl in cal_data}
        ls = self._solver(
            linsolve.quantum.QUBOLogProductSolver,
            cal_data,
            wgts=wgts,
            detrend_phs=True,
            sparse=sparse
        )
        if return_matrix:
            return ls.return_matrix()
        
        prms = ls.solve(mode=mode)
        prms = {self.unpack_sol_key(k): v for k, v in prms.items()}
        sol = RedSol(self.reds, sol_dict=prms)
        # put back in phase trend that was taken out with detrend_phs=True
        for ubl_key in sol.vis:
            sol[ubl_key] *= self.phs_avg[ubl_key].conj()
        if sol0 is not None:
            # put back in sol0 gains that were divided out
            for ant in sol.gains:
                sol.gains[ant] *= sol0.gains.get(ant, 1.0)
        return {}, sol

    def lincal(
        self,
        data,
        sol0,
        wgts={},
        sparse=False,
        mode="qubo",
        conv_crit=1e-10,
        maxiter=50,
        verbose=False,
    ):
        """Taylor expands to linearize redcal equations and iteratively minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary of guess gains and unique model visibilities, keyed by antenna tuples
                like (ant,antpol) or baseline tuples like. Gains should include firstcal gains.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            conv_crit: maximum allowed relative change in solutions to be considered converged
            maxiter: maximum number of lincal iterations allowed before it gives up
            verbose: print stuff
            mode: solving mode passed to the linsolve linear solver ('default', 'lsqr', 'pinv', or 'solve')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().

        Returns:
            meta: dictionary of information about the convergence and chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        sol0pack = {
            self.pack_sol_key(ant): sol0.gains[ant] for ant in self._ants_in_reds
        }
        for ubl in self._ubl_to_reds_index.keys():
            sol0pack[self.pack_sol_key(ubl)] = sol0[ubl]
        ls = self._solver(
            linsolve.quantum.QUBOLinProductSolver,
            self.solver,
            data,
            sol0=sol0pack,
            wgts=wgts,
            sparse=sparse,
        )
        meta, prms = ls.solve_iteratively(
            conv_crit=conv_crit, maxiter=maxiter, verbose=verbose, mode=mode
        )
        prms = {self.unpack_sol_key(k): v for k, v in prms.items()}
        sol = RedSol(self.reds, sol_dict=prms)
        return meta, sol

    def omnical(
        self,
        data,
        sol0,
        wgts={},
        gain=0.3,
        conv_crit=1e-10,
        maxiter=50,
        check_every=4,
        check_after=1,
        wgt_func=lambda x: 1.0,
    ):
        """Use the Liu et al 2010 Omnical algorithm to linearize equations and iteratively minimize chi^2.

        Args:
            solver: a vqls instance
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary of guess gains and unique model visibilities, keyed by antenna tuples
                like (ant,antpol) or baseline tuples like. Gains should include firstcal gains.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            conv_crit: maximum allowed relative change in solutions to be considered converged
            maxiter: maximum number of omnical iterations allowed before it gives up
            check_every: Compute convergence every Nth iteration (saves computation).  Default 4.
            check_after: Start computing convergence only after N iterations.  Default 1.
            gain: The fractional step made toward the new solution each iteration.  Default is 0.3.
                Values in the range 0.1 to 0.5 are generally safe.  Increasing values trade speed
                for stability.
            wgt_func: a function f(abs^2 * wgt) operating on weighted absolute differences between
                data and model that returns an additional data weighting to apply to when calculating
                chisq and updating parameters. Example: lambda x: np.where(x>0, 5*np.tanh(x/5)/x, 1)
                clamps deviations to 5 sigma. Default is no additional weighting (lambda x: 1.).

        Returns:
            meta: dictionary of information about the convergence and chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        sol0pack = {
            self.pack_sol_key(ant): sol0.gains[ant] for ant in self._ants_in_reds
        }
        for ubl in self._ubl_to_reds_index.keys():
            sol0pack[self.pack_sol_key(ubl)] = sol0[ubl]
        ls = self._solver(
            QUBOOmnicalSolver, self.solver, data, sol0=sol0pack, wgts=wgts, gain=gain
        )
        meta, prms = ls.solve_iteratively(
            conv_crit=conv_crit,
            maxiter=maxiter,
            check_every=check_every,
            check_after=check_after,
            wgt_func=wgt_func,
        )
        prms = {self.unpack_sol_key(k): v for k, v in prms.items()}
        sol = RedSol(self.reds, sol_dict=prms)
        return meta, sol

    def count_degens(self, assume_redundant=True):
        """Count the number of degeneracies in this redundant calibrator, given the redundancies and the pol_mode.
        Does not assume coplanarity and instead introduces additional phase slope degeneracies to compensate.

        Args:
            assume_redundant: if True, assume the the array is "redundantly calibrtable" and the only way to get
                extra degneracies is through additional phase slopes (typically 2 per pol for a coplanar array).
                False is slower for large arrays because it has to compute a matrix rank.

        Returns:
            nDegens: the integer number of degeneracies of redundant calibration given the array configuration.
        """
        if assume_redundant:
            nPhaseSlopes = len(
                list(reds_to_antpos(self.reds).values())[0]
            )  # number of phase slope degeneracies
            if self.pol_mode == "1pol":
                return (
                    1 + 1 + nPhaseSlopes
                )  # 1 amplitude degen, 1 phase degen, N phase slopes
            elif self.pol_mode == "2pol":
                return (
                    2 + 2 + 2 * nPhaseSlopes
                )  # 2 amplitude degens, 2 phase degens, 2N phase slopes
            elif self.pol_mode == "4pol":
                return (
                    2 + 2 + nPhaseSlopes
                )  # 4pol ties phase slopes together, so just N phase slopes
            else:  # '4pol_minV'
                return (
                    2 + 1 + nPhaseSlopes
                )  # 4pol_minV ties overall phase together, so just 1 overall phase
        else:
            dummy_data = DataContainer(
                {bl: np.ones((1, 1), dtype=complex) for red in self.reds for bl in red}
            )
            solver = self._solver(
                linsolve.quantum.QuantumLogProductSolver, self.solver, dummy_data
            )
            return np.sum(
                [
                    A.shape[1]
                    - np.linalg.matrix_rank(np.dot(np.squeeze(A).T, np.squeeze(A)))
                    for A in [solver.ls_amp.get_A(), solver.ls_phs.get_A()]
                ]
            )
