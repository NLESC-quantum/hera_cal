#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line driver script for xtalk Filtering with DAYENU that allows for parallelization across baselines."

from hera_cal import xtalk_filter
from hera_cal import vis_clean as vc
import sys

parser = xtalk_filter.xtalk_filter_argparser(mode='dayenu', parallelization_mode='baselines')

a = parser.parse_args()

# set kwargs
filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
baseline_list = vc._parse_baseline_list_string(a.baseline_list)
spw_range = a.spw_range
# Run Delay Filter
delay_filter.load_xtalk_filter_and_write_baseline_list(datafile_list=a.datafile_list, calfile_list=a.calfile_list,
                                         baseline_list=baseline_list, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode='dayenu',
                                         add_to_history=' '.join(sys.argv), **filter_kwargs)
