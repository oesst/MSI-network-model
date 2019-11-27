""" Class to read the result data """

import os
import pickle
import numpy as np
from scipy.signal import find_peaks


class DataReader(object):
    """ Class to read the result data """

    def __init__(self, spatial_conv=False, path='', n_draws=200, readout_time=3950):

        # shall we use dirac simulations
        self.spatial_conv = spatial_conv
        self.n_draws = n_draws
        self.readout_time = readout_time
        self.res_dirs = None
        self.results_dir = None

        # if a path is given take this one, otherwise use cwd
        if len(path) <= 0:
            # get the correct dir
            self.results_dir = os.path.join(os.getcwd(), 'Results')
        else:
            self.results_dir = os.path.join(path, 'Results')

        # Check if we saved the data already
        exp_dir = os.path.join(os.getcwd(), 'Combined_Results')
        # create folder if it does not exists
        if not exp_dir:
            os.mkdir(exp_dir)

        # check which data set to load
        if not self.spatial_conv:
            self.results__combined_file = os.path.join(
                exp_dir, 'results_dirac_.pkl')
        else:
            self.results__combined_file = os.path.join(exp_dir, 'results.pkl')


        # check if results file exists
        if not os.path.exists(self.results__combined_file):

            # choose only dirac simulations
            if not self.spatial_conv:
                self.res_dirs = np.array(
                    [f for f in os.listdir(self.results_dir) if ('Dirac' in f) and not ('test' in f)])
            else:
                self.res_dirs = np.array(
                    [f for f in os.listdir(self.results_dir) if not ('Dirac' in f) and ('Bayesian' in f) and not ('test' in f)])

            # read the data
            self.s_locations, self.s_variances, self.intensities, self.sensory_input, self.fusion = self.read_all_data()
            self.means, self.vars = self.get_means_variances()

            with open(self.results__combined_file, 'wb') as f:
                pickle.dump([self.means, self.vars, self.s_locations, self.s_variances, self.intensities, self.sensory_input, self.fusion], f)
        else:
            # read the data from previously saved file
            with open(self.results__combined_file, 'rb') as f:
                self.means, self.vars, self.s_locations, self.s_variances, self.intensities, self.sensory_input, self.fusion = pickle.load(
                    f)

    def read_all_data(self):
        """ Returns data as follows: loc_a,loc_v, var_a, var_v, int_a, int_v, r_loc_a, r_loc_v, fussion_fb_on, fussion_fb_off """
        # data looks like the following:
        # loc_a,loc_v, var_a, var_v, int_a, int_v, r_loc_a, r_loc_v, fussion
        # where r_loc_a describs the drawn location
        intensities = np.zeros((len(self.res_dirs), self.n_draws, 2))
        s_locations = np.zeros((len(self.res_dirs), self.n_draws, 2))
        s_variances = np.zeros((len(self.res_dirs), self.n_draws, 2))
        sensory_input = np.zeros((len(self.res_dirs), self.n_draws, 2))
        fusion = np.zeros((len(self.res_dirs), self.n_draws, 2))

        for i_sim, e_sim in enumerate(self.res_dirs):

            # read simulation parameters
            results_file = os.path.join(
                self.results_dir, e_sim, 'results.pkl')

            dir_splitted = e_sim.split('_')
            #
            # print(dir_splitted)
            # save intensities
            intensities[i_sim, :, :] = dir_splitted[-6], dir_splitted[-3]
            # save locations
            s_locations[i_sim, :, :] = dir_splitted[-18], dir_splitted[-15]
            # save variances
            s_variances[i_sim, :, :] = dir_splitted[-12], dir_splitted[-9]

            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    net_out, r_all, sensory_input_v, sensory_input_a = pickle.load(
                        f)

                    sensory_input[i_sim, :, 0] = np.argmax(
                        sensory_input_a[:, 1, self.readout_time, :], 1)
                    sensory_input[i_sim, :, 1] = np.argmax(
                        sensory_input_v[:, 1, self.readout_time, :], 1)

                    for i_draw in range(self.n_draws):

                        fusion[i_sim, i_draw, 0] = not (
                            len(find_peaks(np.squeeze(net_out[i_draw, 1, :]), distance=1)[0]) > 1)
                        fusion[i_sim, i_draw, 1] = not (
                            len(find_peaks(np.squeeze(net_out[i_draw, 0, :]), distance=1)[0]) > 1)
        return s_locations, s_variances, intensities, sensory_input, fusion

    def get_means_variances(self):
        """ Returns the mean and variance values of model responses in the following order:
            Model Mean FB On, Model Mean FB Off, Computed Mean
            Model Variance FB On, Model Variance FB Off, Computed Variance"""

        if self.res_dirs is not None:

            # save the means (fb_on, fb_off, computed)
            means = np.zeros((len(self.res_dirs), 3))
            # save the variances (fb_on, fb_off, computed)
            variances = np.zeros((len(self.res_dirs), 3))

            for i_sim, e_sim in enumerate(self.res_dirs):

                # read simulation parameters
                results_file = os.path.join(
                    self.results_dir, e_sim, 'means_vars.pkl')
                if os.path.exists(results_file):
                    with open(results_file, 'rb') as f:
                        res_mean_fb_on, res_var_fb_on, computed_mean, computed_var, res_mean_fb_off, res_var_fb_off = pickle.load(
                            f)
                        means[i_sim, :] = res_mean_fb_on, res_mean_fb_off, computed_mean
                        variances[i_sim,
                                  :] = res_var_fb_on, res_var_fb_off, computed_var
            return means, variances

        else:
            return self.means, self.vars

    def get_s_intensities(self):
        """ Returns intensities for audio and visual stimulis """
        # use the first draw of that data since, the intensity does not change for draws
        return np.squeeze(self.intensities[:, 0, :])

    def get_s_locations(self):
        """ Returns locations for audio and visual stimulis """
        # use the first draw of that data since, the location does not change for draws
        return np.squeeze(self.s_locations[:, 0,:])

    def get_s_variance(self):
        """ Returns variances for audio and visual stimulis """
        # use the first draw of that data since, the variance does not change for draws
        return np.squeeze(self.s_variances[:, 0, :])

    def get_s_sensory_input(self):
        """ Returns the input for audio and visual stimulis """
        # use the first draw of that data since, the variance does not change for draws
        return np.squeeze(self.sensory_input[:, 0, :])

    def get_fusion(self):
        """ Returns variances for audio and visual stimulis """
        # use the first draw of that data since, the variance does not change for draws
        return np.squeeze(self.fusion)
