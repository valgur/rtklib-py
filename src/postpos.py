"""
 post-processing solution from rinex data
"""
from copy import copy, deepcopy

import numpy as np

import __ppk_config as cfg
import rinex as rn
import rtkcmn as gn
from pntpos import pntpos
from rtkpos import rtkinit, rtkpos


def combres(solf, solb):
    # combine forward/backward solutions
    pri = [7, 1, 2, 3, 4, 5, 1, 6]  # priority of solution status
    i, j, solc = 0, len(solb) - 1, []
    gn.trace(3, f'combres: # forward = {len(solf):d}, # backward = {len(solb):d}\n')
    while i < len(solf) or j >= 0:
        if i >= len(solf):
            sol = deepcopy(solb[j])
        elif j < 0:
            sol = deepcopy(solf[i])
        elif not (solf[i].stat == gn.SOLQ_NONE and solb[j].stat == gn.SOLQ_NONE):
            tt = gn.timediff(solf[i].t, solb[j].t)
            if tt < -gn.DTTOL:
                sol = deepcopy(solf[i])
                j += 1
            elif tt > gn.DTTOL:
                sol = deepcopy(solb[j])
                i -= 1
            elif pri[solf[i].stat] < pri[solb[j].stat]:
                sol = deepcopy(solf[i])
            elif pri[solf[i].stat] > pri[solb[j].stat]:
                sol = deepcopy(solb[j])
            else:
                sol = deepcopy(solf[i])
                sol.t = gn.timeadd(sol.t, -tt / 2)
                sol.rr[0:3], sol.qr[0:3, 0:3] = gn.smoother(
                    solf[i].rr[0:3], solb[j].rr[0:3], solf[i].qr, solb[j].qr
                )
            solc.append(sol)
            gn.trace(
                4,
                f'  {sol.t.time:d}: f={solf[i].t.time:d} {solf[i].stat:d} b={solb[j].t.time:d}'
                f' {solb[j].stat:d} tt={tt:.3f}\n',
            )
        i, j = i + 1, j - 1
    return solc


def firstpos(nav, rov, base, dir):
    # find rover position from first obs,
    obsr, obsb = rn.first_obs(nav, rov, base, dir)
    sol = pntpos(obsr, nav)
    # repeat until get solution
    while sol.stat == gn.SOLQ_NONE:
        obsr, obsb = rn.next_obs(nav, rov, base, dir)
        sol = pntpos(obsr, nav)
    gn.trace(3, f'init rr: {sol.rr[0]:.2f} {sol.rr[1]:.2f} {sol.rr[2]:.2f}: {sol.stat:d}\n')
    nav.x[0:6] = copy(sol.rr[0:6])
    nav.rr[0:3] = copy(sol.rr[0:3])


def sqrtvar(cov):
    "sqrt of covariance"
    return np.sqrt(abs(cov)) * np.sign(cov)


def savesol(sol, solfile):
    D2R = gn.rCST.D2R
    solhdr = (
        '%  GPST          latitude(deg) longitude(deg)  height(m)   Q  '
        'ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio\n'
    )
    with open(solfile, 'w') as outfile:
        outfile.write(solhdr)
        for s in sol:
            if s.stat == gn.SOLQ_NONE:
                continue
            wk, sec = gn.time2gpst(s.t)
            llh = gn.ecef2pos(s.rr[0:3])
            std = sqrtvar(gn.covenu(llh, s.qr))
            outfile.write(
                f'{wk:4d} {sec:10.3f} {llh[0] / D2R:14.9f} {llh[1] / D2R:14.9f} {llh[2]:10.4f} '
                f'{s.stat:3d} {s.ns:3d} {std[1, 1]:8.4f}  {std[0, 0]:8.4f} {std[2, 2]:8.4f} '
                f'{std[0, 1]:8.4f} {std[2, 0]:8.4f} {std[1, 2]:8.4f} {s.age:6.2f} {s.ratio:6.1f}\n'
            )


def procpos(nav, rov, base):
    try:
        if nav.filtertype != 'backward':
            # run forward solution
            firstpos(nav, rov, base, dir=1)
            rtkpos(nav, rov, base, dir=1)
            sol0 = deepcopy(nav.sol)
            savesol(sol0, 'forward.pos')
        if nav.filtertype != 'forward':
            # run backward solution
            if nav.filtertype != 'combined_noreset':
                # reset filter states
                rb = nav.rb.copy()
                eph = nav.eph.copy()
                geph = nav.geph.copy()
                glofrq = nav.glofrq.copy()
                maxepoch = nav.maxepoch
                nav = rtkinit(cfg)
                nav.rb = rb
                nav.eph, nav.geph = eph, geph
                nav.glofrq = glofrq
                nav.maxepoch = maxepoch
                firstpos(nav, rov, base, dir=-1)
            else:  # combined_noreset
                nav.sol = [nav.sol[-1]]
            rtkpos(nav, rov, base, dir=-1)
            savesol(nav.sol, 'backward.pos')
        if nav.filtertype == 'combined' or nav.filtertype == 'combined_noreset':
            sol = combres(sol0, nav.sol)
            savesol(sol, 'combined.pos')
            return sol
    except KeyboardInterrupt:
        pass
    return nav.sol
