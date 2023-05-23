#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:40:20 2023

@author: nugenta
"""

from mne.preprocessing import ICA
from mne.preprocessing import corrmap
import glob, os, os.path as op
import mne
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.pyplot import show
import numpy as np
import warnings
import unittest.mock
from mne.defaults import _INTERPOLATION_DEFAULT, _EXTRAPOLATE_DEFAULT, _BORDER_DEFAULT, _handle_default
from mne.utils import logger
import mne.viz.topomap
import mne.viz
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _prepare_topomap, _setup_interp, _get_patch, _draw_outlines
from mne.viz.utils import _setup_vmin_vmax, _get_cmap, plt_show
from scipy.io import savemat
import PIL.Image


import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import interpolate
from scipy.io import savemat
from scipy.stats import zscore

from MEGnet.prep_inputs.ICA import cart2sph, pol2cart, cart2pol, sensor_pos2circle
from MEGnet.prep_inputs.ICA import circle_plot

import sys

import matplotlib
matplotlib.use('agg')
# plt.ion()


topdir = sys.argv[1]
# set up directories and files
# topdir='/Volumes/EnigmaMeg/BIDS/ICA/ICA_megnet/'
# topdir='/fast/MEGNET/ICA_megnet/RAW/HCP_reprocessed_newtopo'
# outdir='/fast/MEGNET/HCP_megnet_testing'



# =============================================================================
# Helper Functions
# =============================================================================

# re-write of the MNE python function make_head_outlines() without the nose and ears, with expansion 
# of the outline to 1.01 

def make_head_outlines_new(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius*1.01 + x
    head_y = np.sin(ll) * radius*1.01 + y
    dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
    dx, dy = dx.real, dx.imag
    
    outlines_dict = dict(head=(head_x, head_y))
    
    # Make the figure encompass slightly more than all points
    mask_scale = 1.
    # We probably want to ensure it always contains our most
    # extremely positioned channels, so we do:
    mask_scale = max(
            mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
    
    outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
    clip_radius = radius * mask_scale
    outlines_dict['clip_radius'] = (clip_radius,) * 2
    outlines_dict['clip_origin'] = clip_origin      
    
    return outlines_dict

def _mod_plot_topomap(
    data,
    pos,
    axes,
    *,
    ch_type="eeg",
    sensors=True,
    names=None,
    mask=None,
    mask_params=None,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    cmap=None,
    vmin=None,
    vmax=None,
    cnorm=None,
    show=True,
    onselect=None,
):
    from matplotlib.colors import Normalize
    from matplotlib.widgets import RectangleSelector

    data = np.asarray(data)
    logger.debug(f"Plotting topomap for {ch_type} data shape {data.shape}")

    if isinstance(pos, mne.Info):  # infer pos from Info object
        picks = _pick_data_channels(pos, exclude=())  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = _get_channel_types(pos, unique=True)
        info_help = (
            "Pick Info with e.g. mne.pick_info and "
            "mne.io.pick.channel_indices_by_type."
        )
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " + info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError(
                "Number of channels in the Info object (%s) and "
                "the data array (%s) do not match. " % (len(pos["chs"]), data.shape[0])
                + info_help
            )
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ("planar", "grad")):
            # deal with grad pairs
            picks = _pair_grad_sensors(pos, topomap_coords=False)
            pos = _find_topomap_coords(pos, picks=picks[::2], sphere=sphere)
            data, _ = _merge_ch_data(data[picks], ch_type, [])
            data = data.reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)

    extrapolate = _check_extrapolate(extrapolate, ch_type)
    if data.ndim > 1:
        raise ValueError(
            "Data needs to be array of shape (n_sensors,); got "
            "shape %s." % str(data.shape)
        )

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = (
        "Electrode positions should be specified as a 2D array with "
        "shape (n_channels, 2). Each row in this matrix contains the "
        "(x, y) position of an electrode."
    )
    if pos.ndim != 2:
        error = (
            "{ndim}D array supplied as electrode positions, where a 2D "
            "array was expected"
        ).format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = (
            "The supplied electrode positions matrix contains 3 columns. "
            "Are you trying to specify XYZ coordinates? Perhaps the "
            "mne.channels.create_eeg_layout function is useful for you."
        )
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)
    pos = pos[:, :2]

    if len(data) != len(pos):
        raise ValueError(
            "Data and pos need to be of same length. Got data of "
            "length %s, pos of length %s" % (len(data), len(pos))
        )

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = "Reds" if norm else "RdBu_r"
    cmap = _get_cmap(cmap)

    outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    assert isinstance(outlines, dict)

    _prepare_topomap(pos, axes)

    mask_params = _handle_default("mask_params", mask_params)

    # find mask limits and setup interpolation
    extent, Xi, Yi, interp = _setup_interp(
        pos, res, image_interp, extrapolate, outlines, border
    )
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # plot outline
    patch_ = _get_patch(outlines, extrapolate, interp, axes)

    # get colormap normalization
    if cnorm is None:
        cnorm = Normalize(vmin=vmin, vmax=vmax)

    # plot interpolated map
    if image_interp == "nearest":  # plot over with Voronoi, more accurate
        im = _voronoi_topomap(
            data,
            pos=pos,
            outlines=outlines,
            ax=axes,
            cmap=cmap,
            norm=cnorm,
            extent=extent,
            res=res,
        )
    else:
        im = axes.imshow(
            Zi,
            cmap=cmap,
            origin="lower",
            aspect="equal",
            extent=extent,
            interpolation="bilinear",
            norm=cnorm,
            alpha=0.9
        )

    # gh-1432 had a workaround for no contours here, but we'll remove it
    # because mpl has probably fixed it
    linewidth = mask_params["markeredgewidth"]
    cont = True
    if isinstance(contours, (np.ndarray, list)):
        pass
    elif contours == 0 or ((Zi == Zi[0, 0]) | np.isnan(Zi)).all():
        cont = None  # can't make contours for constant-valued functions
    if cont:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            cont = axes.contour(
                Xi, Yi, Zi, contours, colors="w", linewidths=linewidth / 2.0, negative_linestyles='solid', alpha=0.5, antialiased=True
            )
    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T
    mask = mask.astype(bool, copy=False) if mask is not None else None
    if sensors is not False and mask is None:
        _topomap_plot_sensors(pos_x, pos_y, sensors=sensors, ax=axes)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        axes.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _topomap_plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=axes)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        axes.plot(pos_x[idx], pos_y[idx], **mask_params)

    if isinstance(outlines, dict):
        _draw_outlines(axes, outlines)

    if names is not None:
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (_pos, _name) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            axes.text(
                _pos[0],
                _pos[1],
                _name,
                horizontalalignment="center",
                verticalalignment="center",
                size="x-small",
            )

    if not axes.figure.get_constrained_layout():
        axes.figure.subplots_adjust(top=0.95)

    if onselect is not None:
        lim = axes.dataLim
        x0, y0, width, height = lim.x0, lim.y0, lim.width, lim.height
        axes.RS = RectangleSelector(axes, onselect=onselect)
        axes.set(xlim=[x0, x0 + width], ylim=[y0, y0 + height])
    plt_show(show)
    return im, cont, interp

#Monkey patch the new topography output
mne.viz.topomap._plot_topomap=_mod_plot_topomap


# =============================================================================
# 
# =============================================================================



def make_topos_datarun(folder, outdir=None):
    ica=mne.preprocessing.read_ica(f'{folder}/{folder}_0-ica.fif')
    raw=mne.io.read_raw_fif(f'{folder}/{folder}_250srate_meg.fif')
    
    # pick only the MEG channels, and then grab the indices of the channels we want to use 
    raw.pick_types(meg=True, eeg=False, ref_meg=False)
    mag_idxs = mne.pick_types(raw.info, ref_meg=False, meg='mag')
    
    # generate the sensor positions on the circle
    circle_pos = sensor_pos2circle(raw, ica)
    
    for comp in range(0,20):
      # get the ICA component data
      data=np.dot(ica.mixing_matrix_[:,comp].T, ica.pca_components_[:ica.n_components_])
      data = data[mag_idxs]
    
      '''Generate the plot and save'''
      # create a circular outline without nose and ears, and get coordinates
      outlines_new = make_head_outlines_new(np.array([0,0,0,1]),
                                              circle_pos,
                                              'head',
                                              (0,0))
      # set up the figure canvas
      # fig = plt.figure(figsize=(1.5, 1.8), dpi=100, facecolor='black')
      fig = plt.figure(figsize=(1.3, 1.3), dpi=100, facecolor='black')
      canvas=FigureCanvas(fig)
      ax = fig.add_subplot(111)
      
      # plot the figure using the monkey patched mne.viz.plot_topomap
      mnefig, contour = mne.viz.plot_topomap(data,
                                    circle_pos[mag_idxs],
                                    sensors=False,
                                    outlines=outlines_new,
                                    extrapolate='head',
                                    sphere=[0,0,0,1.],
                                    contours=10,res=120,
                                    show=True,
                                    axes=ax, 
                                    cmap='bwr'
                                    )
      
      print(comp)
      # plt.show()
    
      # So matplotlib did strange things to the canvas and background image size when I tried to grab
      # the image from the canvas and convert directly to RGB. This ridiculous (but functional) hack is to 
      # write the image out as a png then reopen it to convert to RGB. 
      outpng = f'{outdir}/component{str(int(comp)+1)}.png'
      mnefig.figure.savefig(outpng,dpi=120,bbox_inches='tight',pad_inches=0)
      rgba_image=PIL.Image.open(outpng)
      rgb_image=rgba_image.convert('RGB')
      
      # # mat_fname = os.path.splitext(out_fname)[0]+'.mat'
      # matrix_out = np.frombuffer(mnefig.figure.canvas.tostring_rgb(), dtype=np.uint8)
      # matrix_out = matrix_out.reshape(mnefig.figure.canvas.get_width_height()[::-1] + (3,))
    
      # save the RGB image as a .mat file
      mat_fname = f'{outdir}/component{str(int(comp)+1)}.mat'
      savemat(mat_fname, {'array':np.array(rgb_image)})
      del mnefig

os.chdir(topdir)
folders = glob.glob(op.join(topdir,'*'))
folders = [op.basename(i) for i in folders]

for folder in folders:
    make_topos_datarun(folder, outdir=op.join(topdir, folder))







