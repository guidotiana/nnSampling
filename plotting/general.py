import numpy as np


# Share axs limits
def share_axs(axs, which_ax='y', mag=1.):
    assert which_ax in ('x', 'y'), f'share_axs(): invalid value for variable <which_ax> (found {which_ax}). Allowed values: "x", "y". Exit!'
    assert mag >= 1., f'share_axs(): invalid value for variable <mag> (found {mag}). Allowed values: mag >= 1. Exit!'
    nrows, ncols = axs.shape
    if which_ax == 'y':
        for irow in range(nrows):
            ylim = [np.inf, -np.inf]
            for icol in range(ncols):
                col_ylim = axs[irow, icol].get_ylim()
                if col_ylim[0] < ylim[0]: ylim[0] = col_ylim[0]
                if col_ylim[1] > ylim[1]: ylim[1] = col_ylim[1]
            if not isclose(mag, 1.):
                ymean = (ylim[0]+ylim[1])/2.
                mag_yrange = mag*(ylim[1]-ylim[0])
                ylim = [ymean-mag_yrange/2., ymean+mag_yrange/2.]
            for icol in range(ncols):
                axs[irow, icol].set_ylim(ylim)
    else:
        for icol in range(ncols):
            xlim = [np.inf, -np.inf]
            for irow in range(nrows):
                row_xlim = axs[irow, icol].get_xlim()
                if row_xlim[0] < xlim[0]: xlim[0] = row_xlim[0]
                if row_xlim[1] > xlim[1]: xlim[1] = row_xlim[1]
            if not isclose(mag, 1.):
                xmean = (xlim[0]+xlim[1])/2.
                mag_xrange = mag*(ylim[1]-ylim[0])
                xlim = [xmean-mag_xrange/2., xmean+mag_xrange/2.]
            for irow in range(nrows):
                axs[irow, icol].set_xlim(xlim)
    return axs


# Create 2-D histogram profile line
def get_hist_profile(counts, bins, hide_zeros=False):
    profile_x, profile_y = [], []
    for ibin in range(len(counts)):
        if ibin == len(counts)-1: break
        profile_x.append(bins[ibin])
        profile_y.append(counts[ibin])
        if not isclose(counts[ibin+1], counts[ibin]):
            profile_x.append(bins[ibin+1])
            profile_y.append(counts[ibin])
    if counts[-1] > 0:
        profile_x = np.array(profile_x + [bins[-2], bins[-1], bins[-1]])
        profile_y = np.array(profile_y + [counts[-1], counts[-1], 0])
    else:
        profile_x = np.array(profile_x + list(bins[-2:]))
        profile_y = np.array(profile_y + [counts[-1], 0])
    if hide_zeros:
        fidx, lidx = 0, len(profile_y)-1
        while profile_y[fidx+1] == 0: fidx += 1
        while profile_y[lidx-1] == 0: lidx -= 1
        profile_x = profile_x[fidx:lidx+1]
        profile_y = profile_y[fidx:lidx+1]
    return profile_x, profile_y
