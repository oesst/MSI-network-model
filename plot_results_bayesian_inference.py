""" Here we readout all calculated means anv vars from different simulations """
# import matplotlib
# matplotlib.use('Qt5Agg')
import data_reader
import plotter_bayesian_data
import numpy as np
import pickle
import os

# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from importlib import reload  # Python 3.4+ only.
# reload(plotter_bayesian_data)
# plter = plotter_bayesian_data.Plotter('./', save_figs=False)

# choose which data should be used
no_spatial_conv = True
n_draws = 200

plter = plotter_bayesian_data.Plotter('./', save_figs=False)

############### READ DATA ###############
reader = data_reader.DataReader(
        not no_spatial_conv, path='/media/oesst/TOSHIBA EXT/Backup PHD Stuff/Code/Python/msi_network_model')

# get the means and variances of the results
means, vars = reader.get_means_variances()
intensities = reader.get_s_intensities()
s_locations = reader.get_s_locations()
s_variances = reader.get_s_variance()


# caluclate the percentage of fused event in a simulation
fusion = reader.get_fusion()
fusion_fb_on = fusion[:,:,0]
fusion_fb_off = fusion[:,:,1]

# # %% Save the data for faster excess
# exp_dir = os.path.join(os.getcwd(), 'Combined_Results')
#
# if not no_spatial_conv:
#     results_file = os.path.join(exp_dir, 'results.pkl')
# else:
#     results_file = os.path.join(exp_dir, 'results_dirac.pkl')
#
# if not os.path.exists(results_file):
#
#     # data reader for reading data
#     reader = data_reader.DataReader(
#         not no_spatial_conv, path='/media/oesst/TOSHIBA EXT/Backup PHD Stuff/Code/Python/msi_network_model')
#
#     # get the means and variances of the results
#     means, vars = reader.get_means_variances()
#     intensities = reader.get_s_intensities()
#     s_locations = reader.get_s_locations()
#     s_variances = reader.get_s_variance()
#
#     data_all = reader.get_all_data()
#
#     # caluclate the percentage of fused event in a simulation
#     fusion_fb_on = data_all[:, :, -2]
#     fusion_fb_off = data_all[:, :, -1]
#
#     with open(results_file, 'wb') as f:
#         pickle.dump([means, vars, intensities, s_locations,
#                      s_variances, fusion_fb_on, fusion_fb_off], f)
# else:
#     with open(results_file, 'rb') as f:
#         means, vars, intensities, s_locations, s_variances, fusion_fb_on, fusion_fb_off = pickle.load(
#             f)

fusion_fb_on_perc = 100 * \
    (np.count_nonzero(fusion_fb_on, 1) / n_draws + np.finfo(float).eps)

fusion_fb_off_perc = 100 * \
    (np.count_nonzero(fusion_fb_off, 1) / n_draws + np.finfo(float).eps)
spatial_offsets = s_locations[:, 0] - s_locations[:, 1]

# %%


# data looks like the following:
# loc_a,loc_v, var_a, var_v, int_a, int_v, r_loc_a, r_loc_v, fussion_fb_on_ fussion_fb_off


############### PLOTTING ###############


# %% Plot Fusion Probability over Spatial Offset

ax = plter.plot_fusion_prob(
    spatial_offsets, fusion_fb_on_perc, fusion_fb_off_perc, std_plot=True)


# %% Fit Sigmoid to Fusion Probability over Spatial Offset

ax = plter.plot_fitted_sigmoid(
    spatial_offsets, fusion_fb_on_perc, fusion_fb_off_perc)

# %% Plots the computed mean and var values against the model reponse

ax = plter.plot_mean_var_comparison(means, vars)


# %%
ax = plter.plot_intens_var_comparison(intensities, vars, lin_reg=True)

# %%
ax = plter.plot_location_var_comparison(s_locations, vars, lin_reg=True)

# %%
ax = plter.plot_var_var_comparison(s_variances, vars, lin_reg=True)


#
# spatial_offsets.shape
s_variances[np.where(np.logical_and(spatial_offsets == 6.5 , np.abs(vars[:, -1] - vars[:, 1]) >2 ))]
intensities[np.where(np.logical_and(spatial_offsets == 6.5 , np.abs(vars[:, -1] - vars[:, 1]) >2 ))]
s_locations[np.where(np.logical_and(spatial_offsets == 6.5 , np.abs(vars[:, -1] - vars[:, 1]) >2 ))]

# %%
# fig = plt.figure(figsize=(10, 10))
# var_error_cor_off = np.abs(vars[:, -1] - vars[:, 1])
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(s_variances[:, 0] - s_variances[:, 1],
#            s_locations[:, 0] - s_locations[:, 1], fusion_fb_on_perc)
# plt.show()
