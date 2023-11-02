import argparse
from copy import deepcopy 

import numpy as np
from hera_sim.antpos import hex_array, linear_array
from hera_sim.vis import sim_red_data
# from hera_cal import redcal as om
from hera_cal import qubo_redcal as om 
from hera_cal.utils import split_pol
from hera_sim.sigchain import Bandpass
from hera_sim.foregrounds import DiffuseForeground, PointSourceForeground
import matplotlib.pyplot as plt 
from scipy.stats import kde, chi2

from hera_cal.datacontainer import DataContainer, RedDataContainer
from hera_cal.noise import infer_dt, predict_noise_variance_from_autos
from hera_cal.utils import split_bl
import h5py 
import hdfdict

from qalcore.dwave.qubols.qubols import QUBOLS
from qalcore.dwave.qubols.encodings import RealQbitEncoding, RealUnitQbitEncoding, EfficientEncoding


def save_system_data(filename, antpos, data, freqs, lsts):

    # change the pos keys
    pos = deepcopy(antpos)
    keys = list(pos.keys())
    for k in keys:
        pos[str(k)] = pos.pop(k)

    with h5py.File(filename, 'w') as f5:
        f5.create_group('antpos')
        hdfdict.dump(pos, f5['antpos'])

        f5.create_group('data')
        hdfdict.dump(data, f5['data'])

        f5.create_dataset('freqs', data=freqs)
        f5.create_dataset('lsts', data=lsts)

def save_sol_data(filename, sol_fc=None, sol_logcal=None, sol_omnical=None):
    # save to hdf5
    with h5py.File(filename,'w') as h5f:
        if sol_fc is not None:
            h5f.create_group('firstcal')
            hdfdict.dump(sol_fc, h5f['firstcal'])

        if sol_logcal is not None:
            h5f.create_group('logcal')
            hdfdict.dump(sol_logcal, h5f['logcal'])

        if sol_omnical is not None:
            h5f.create_group('omnical')
            hdfdict.dump(sol_omnical, h5f['omnical'])

def save_solver_data(filename, solver_fc=None, solver_logcal=None, solver_omnical=None):
    # save to hdf5
    with h5py.File('solver_' + filename,'w') as h5f:
        if solver_fc is not None:
            h5f.create_group('firstcal')
            hdfdict.dump(solver_fc.todict(), h5f['firstcal'])

        if solver_logcal is not None:
            h5f.create_group('logcal')
            hdfdict.dump(solver_logcal.todict(), h5f['logcal'])

        if solver_omnical is not None:
            h5f.create_group('omnical')
            hdfdict.dump(solver_omnical.todict(), h5f['omnical'])

def save_meta_data(filename, meta_fc=None, meta_logcal=None, meta_omnical=None):
    # save to hdf5
    with h5py.File('meta_' + filename,'w') as h5f:
        if meta_fc is not None:
            h5f.create_group('firstcal')
            hdfdict.dump(meta_fc, h5f['firstcal'])

        if meta_logcal is not None:
            h5f.create_group('logcal')
            hdfdict.dump(meta_logcal, h5f['logcal'])

        if meta_omnical is not None:
            h5f.create_group('omnical')
            hdfdict.dump(meta_omnical, h5f['omnical'])

def setup_system(args):

    # positions
    antpos = hex_array(args.num_rings, split_core=args.split_core, outriggers=args.outriggers)
    
    # redundant baselines
    reds = om.get_reds(antpos, pols=['xx'])

    # Define frequencies 
    freqs = np.linspace(120e6, 180e6, args.nfreqs) # in Hz
    df = np.median(np.diff(freqs))
    # define the Local Sidereal Times
    times = np.linspace(0, 600. / 60 / 60 / 24, args.ntimes, endpoint=False)
    dt = np.median(np.diff(times)) * 3600. * 24

    # Simulate redundant data with noise
    noise_var = args.noise
    gains, true_viz, data = sim_red_data(reds, shape=(len(times), len(freqs)), gain_scatter=.1)
    ants = gains.keys()
    noise = DataContainer({bl: np.sqrt(noise_var / 2) * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape)) for bl, vis in data.items()})
    noisy_data = noise + DataContainer(data)

    # Set up autocorrelations so that the predicted noise variance is the actual simulated noise variance
    for antnum in antpos.keys():
        noisy_data[(antnum, antnum, 'xx')] = np.ones((len(times), len(freqs))) * np.sqrt(noise_var * dt * df)
    noisy_data.freqs = deepcopy(freqs)
    noisy_data.times_by_bl = {bl[0:2]: deepcopy(times) for bl in noisy_data.keys()}

    return antpos, reds, noisy_data, freqs, times

def setup_system_point_source(args):

    # antenna positions
    antpos = hex_array(args.num_rings, split_core=args.split_core, outriggers=args.outriggers)

    # redundancy in the array
    reds = om.get_reds(antpos, pols=['ee'], pol_mode='1pol')

    # Define frequencies 
    freqs = np.linspace(120e6, 180e6, args.nfreqs) # in Hz

    # define the Local Sidereal Times
    lsts = np.linspace(0, 600. / 60 / 60 / 24, args.ntimes, endpoint=False)

    # Gains with a bandpass model and cable delays bandpass with cable delays
    bp = Bandpass(gain_spread=.1, dly_rng=(-20, 20))
    gains = {(ant, 'Jee'): gain[None, :] for ant, gain in bp(freqs / 1e9, list(antpos.keys())).items()}

    # visibilities with point source foreground
    vis = {red[0]: PointSourceForeground()(lsts, freqs / 1e9,  antpos[red[0][1]] - antpos[red[0][0]]) for red in reds}

    # Build RedSol object with true solutions
    sol_true = om.RedSol(reds, gains=gains, vis=vis)

    # Create uncalibrated data
    data = {bl: sol_true[red[0]] * sol_true[bl[0], 'Jee'] * sol_true[bl[1], 'Jee'].conj() for red in reds for bl in red}

    # Simulate redundant data with noise
    noise_var = args.noise
    noise = DataContainer({bl: np.sqrt(noise_var / 2) * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape)) for bl, vis in data.items()})
    noisy_data = noise + DataContainer(data)

    return antpos, reds, noisy_data, freqs, lsts

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Options for redcal jobs")
    parser.add_argument("--seed", default=21, help="Seed for random number generator", type=int)
    parser.add_argument("--num_rings", default=2, help="Number of hex rings", type=int)
    parser.add_argument("--split_core", default=False, action="store_true", help="split the hex core")
    parser.add_argument("--outriggers", default=0, help="number of outriggers", type=int)
    parser.add_argument("--system_filename", default='system.h5', help="name of the system file", type=str)
    parser.add_argument("--output_filename", default='redcal.h5', help="name of the output file", type=str)
    parser.add_argument("--nfreqs", default=100, help="number of frequencies", type=int)
    parser.add_argument("--ntimes", default=256, help="number of times", type=int)
    parser.add_argument("--noise", default=1E-1, help="Noise level", type=float)

    
    args = parser.parse_args()
    np.random.seed(args.seed) 

    # setup system
    # antpos, reds, noisy_data, freqs, lsts = setup_system_point_source(args)
    antpos, reds, noisy_data, freqs, lsts = setup_system(args)
    save_system_data(args.system_filename, antpos, noisy_data, freqs, lsts)

    # define the ansatz
    options = {'num_reads':100, 'num_qbits':11, 'encoding':EfficientEncoding}
    solver = QUBOLS(options)

    
    # calibrate
    rc = om.QUBORedundantCalibrator(reds, solver=solver)
    meta_fc, sol_fc, solver_data_fc = rc.firstcal(noisy_data, freqs)
    # meta_logcal, sol_logcal = rc.logcal(noisy_data, sol0=sol_fc)
    # meta_omnical, sol_omnical = rc.omnical(noisy_data, sol_logcal, maxiter=500, check_after=50, check_every=10)
    # sol_omnical.remove_degen(degen_sol=sol_fc, inplace=True)

    # save to hdf5
    save_sol_data(args.output_filename, sol_fc=sol_fc)
    save_meta_data(args.output_filename, meta_fc=meta_fc)
    save_solver_data(args.output_filename, solver_fc=solver_data_fc)

