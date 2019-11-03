""" This is the main file for a MSI model experiment """

import numpy as np
import plotter
import network
import os
import pickle

# %%markdown
#
# %%

# Use a unique name for each experiments
exp_name = 'Spatial_Principle'

########################################
######## Experiment PARAMETERS #########

# set number of neurons in
n_neurons_msi = 20

# set number of input intensity, all inputs need to have the same amount of intensities
intensity_tests = 11
sensory_intensities = np.linspace(0, 1, intensity_tests)
# define the sensory input properties (reliability of sensory signal is encoded in intensity and sigma)
intensity_s_v = sensory_intensities
intensity_s_a = sensory_intensities
intensity_c_v = sensory_intensities
intensity_c_a = sensory_intensities

# Which conditions to test:
#  0 : both sensory stimuli, no cortical feedback
#  1 : both sensory stimuli, both cortical inputs
#  2 : both sensory stimuli, no cortical visual input
#  3 : both sensory stimuli, no cortical auditory input
#  4 : all auditory input (sensory, cortical), no visual
#  5 : all visual input (sensory, cortical), no auditory
#  6 : only auditory sensory input, both cortical
conditions_to_test = [0, 1, 2, 3, 4, 5, 6]

# defines the stimuli with location (x,y), onset, duration
s_onset_temp = 10
s_onset_spatial = 8
s_spatial_offsets = np.arange(0, 12, 2)


# define the uncertaintiy of the inputs
sigma_s_v = 2
sigma_s_a = 2
sigma_c_v = 2
sigma_c_a = 2

readout_time = 3950


# Create the network and initialize all internal vars
net = network.Network(exp_name, n_neurons_msi=n_neurons_msi)

# %% Create directory according to exp name

# create directory for results if it doesnt extist
exp_name_neurons = exp_name + '_neurons_' + str(n_neurons_msi) + '_sigmas_' + str(sigma_s_v) + str(sigma_s_a) + str(
    sigma_c_v) + str(sigma_c_a) + '_onset_' + str(s_onset_spatial) + '_offsets_' + str(len(s_spatial_offsets)) + '_intensities_' + str(intensity_tests)
exp_dir = path = os.path.join(os.getcwd(), 'Results')
# create result directory if it doesnt exists
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

exp_dir = path = os.path.join(exp_dir, exp_name_neurons)

# check if directory exists and if its not empty
if os.path.exists(exp_dir) and os.listdir(exp_dir):
    print('Directory ' + exp_dir +
          ' already exists. Using existing data and results...')
    skip_simulation = True
else:
    skip_simulation = False
    print('Creating directory : ', exp_dir)
    os.mkdir(exp_dir)

    # create a file with all parameters
    with open(os.path.join(exp_dir, exp_name_neurons), 'w+') as f:  # Python 3: open(..., 'wb')
        f.write(exp_name + '\n \n')
        f.write('Sensory Intensity Audio  : ' + str(intensity_s_a) + '\n')
        f.write('Sensory Intensity Video  : ' + str(intensity_s_v) + '\n')
        f.write('Cortical Intensity Audio : ' + str(intensity_c_a) + '\n')
        f.write('Cortical Intensity Video : ' + str(intensity_c_v) + '\n')
        f.write('\n \n')
        f.write('Sensory Audio Sigma  : ' + str(sigma_s_a) + '\n')
        f.write('Sensory Video Sigma  : ' + str(sigma_s_v) + '\n')
        f.write('Cortical Audio Sigma : ' + str(sigma_c_a) + '\n')
        f.write('Cortical Video Sigma : ' + str(sigma_c_v) + '\n')
        f.write('\n \n')
        f.write('Conditions to test : ' + str(conditions_to_test) + '\n')
        f.write('\n \n')
        f.write('Spatial Onset : ' + str(s_onset_spatial) + '\n')
        f.write('Spatial Offset : ' + str(s_spatial_offsets) + '\n')
        f.close()


# %%
########################################
# initiualize arrays

net_out = np.zeros((len(s_spatial_offsets), intensity_tests,
                    len(conditions_to_test), net.n_neurons_msi))
sensory_input_v = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
sensory_input_a = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
cortical_input_v = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
cortical_input_a = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
r_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
p_pool_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
p_sensory_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
q_fb_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
q_s2_v_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
q_s2_a_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
q_s1_v_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))
q_s1_a_all = np.zeros((len(s_spatial_offsets), intensity_tests, len(
    conditions_to_test), net.len_t, net.n_neurons_msi))

if not skip_simulation:
    for i_offset in range(len(s_spatial_offsets)):
        for i_intens in range(intensity_tests):
            for i_condi, condi in enumerate(conditions_to_test):

                ########################
                # Create Input Stimuli #
                ########################

                # sensory inputs
                stimuli_s_v = {'loc': np.array([s_onset_spatial + s_spatial_offsets[i_offset]]), 'onset': s_onset_temp,
                               'duration': net.len_t - s_onset_temp, 'sigma': sigma_s_v, 'weight': intensity_s_v[i_intens]}
                stimuli_s_a = {'loc': np.array([s_onset_spatial]), 'onset': s_onset_temp,
                               'duration': net.len_t - s_onset_temp, 'sigma': sigma_s_a, 'weight': intensity_s_a[i_intens]}

                # cortical inputs
                stimuli_c_v = {'loc': np.array([s_onset_spatial + s_spatial_offsets[i_offset]]), 'onset': s_onset_temp,
                               'duration': net.len_t - s_onset_temp, 'sigma': sigma_c_v, 'weight': intensity_c_v[i_intens]}
                stimuli_c_a = {'loc': np.array([s_onset_spatial]), 'onset': s_onset_temp,
                               'duration': net.len_t - s_onset_temp, 'sigma': sigma_c_a, 'weight': intensity_c_a[i_intens]}

                # Create inputs
                sens_in_v, sens_in_a, cor_in_v, cor_in_a = net.create_inputs(
                    stimuli_s_v, stimuli_s_a, stimuli_c_v, stimuli_c_a, gauss=True)

                # run the network with random locations
                r, act, p_pool, p_sensory, q_fb, q_s2_v, q_s2_a, q_s1_v, q_s1_a = net.run(
                    i_condi)

                # save the data
                net_out[i_offset, i_intens, i_condi, :] = act[readout_time, :]
                r_all[i_offset, i_intens, i_condi, :, :] = r
                p_pool_all[i_offset, i_intens, i_condi, :, :] = p_pool
                p_sensory_all[i_offset, i_intens, i_condi, :, :] = p_sensory
                q_fb_all[i_offset, i_intens, i_condi, :, :] = q_fb
                q_s2_v_all[i_offset, i_intens, i_condi, :, :] = q_s2_v
                q_s2_a_all[i_offset, i_intens, i_condi, :, :] = q_s2_a
                q_s1_v_all[i_offset, i_intens, i_condi, :, :] = q_s1_v
                q_s1_a_all[i_offset, i_intens, i_condi, :, :] = q_s1_a
                sensory_input_v[i_offset, i_intens, i_condi, :, :] = sens_in_v
                sensory_input_a[i_offset, i_intens, i_condi, :, :] = sens_in_a
                cortical_input_v[i_offset, i_intens, i_condi, :, :] = cor_in_v
                cortical_input_a[i_offset, i_intens, i_condi, :, :] = cor_in_a

                print('Spatial Offset ' + str(i_offset + 1) + ' of ' + str(int(len(s_spatial_offsets))) + '. Intensity Test ' + str(i_intens +
                                                                                                                                    1) + ' of ' + str(int(intensity_tests)) + '. Condition : ' + str(i_condi + 1) + ' of ' + str(len(conditions_to_test)))
###### Save outputs ######

results_file = os.path.join(exp_dir, 'results.pkl')
if not os.path.exists(results_file):
    with open(results_file, 'wb') as f:
        pickle.dump([net_out, r_all, p_pool_all, p_sensory_all, q_fb_all, q_s2_v_all, q_s2_a_all,
                     q_s1_v_all, q_s1_a_all, sensory_input_v, sensory_input_a, cortical_input_v, cortical_input_a], f)
else:
    with open(results_file, 'rb') as f:
        net_out, r_all, p_pool_all, p_sensory_all, q_fb_all, q_s2_v_all, q_s2_a_all, q_s1_v_all, q_s1_a_all, sensory_input_v, sensory_input_a, cortical_input_v, cortical_input_a = pickle.load(
            f)


###### Plotting ######
# %%

condition = 1
inten = 3

plter = plotter.Plotter(exp_dir, save_figs=False)
ax = plter.plot_normal_response(net_out, s_onset_spatial, sensory_input_v[:, inten, condition, readout_time, :], sensory_input_a[:,
                                                                                                       inten, condition, readout_time, :], s_spatial_offsets, sensory_intensities, conditions_to_test)
