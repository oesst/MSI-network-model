""" This simulation is adapted from main for Bayesian inference analysis """

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import plotter
import network
import os
import pickle
import numpy as np

# %%markdown
#
# %%

# do not use spatial convolution (set kernels supe small)
no_spatial_conv = True
# Use a unique name for each experiments
exp_name = 'Bayesian_Inference_Mod_test'
if no_spatial_conv:
    exp_name += '_Dirac_Kernels'

########################################
######## Experiment PARAMETERS #########

# set number of neurons in
n_neurons_msi = 20


# Which conditions to test:
#  0 : both sensory stimuli, no cortical feedback
#  1 : both sensory stimuli, both cortical inputs
#  2 : both sensory stimuli, no cortical visual input
#  3 : both sensory stimuli, no cortical auditory input
#  4 : all auditory input (sensory, cortical), no visual
#  5 : all visual input (sensory, cortical), no auditory
#  6 : only auditory sensory input, both cortical
conditions_to_test = [0, 1]

# defines the stimuli with location (x,y), onset, duration

s_a_mean = 10
s_v_mean = 8
s_a_var = 1.5
s_v_var = 1.0
s_a_intensity = 0.5
s_v_intensity = 0.5


s_onset_temp = 0


# define the uncertaintiy of the inputs
sigma_s_v = 2
sigma_s_a = 2
sigma_c_v = 2
sigma_c_a = 2

readout_time = 3950

# define how many times we draw from the distribution
n_draws = 3

# Create the network and initialize all internal vars
net = network.Network(exp_name, n_neurons_msi=n_neurons_msi)

# %% Create directory according to exp name

# create directory for results if it doesnt extist
exp_name_neurons = exp_name + '_neurons_' + str(n_neurons_msi) + '_sigmas_' + str(sigma_s_v) + str(sigma_s_a) + str(
    sigma_c_v) + str(sigma_c_a) + '_mean_a_' + str(s_a_mean) + '_mean_v_' + str(s_v_mean) + '_var_a_' + str(s_a_var) + '_var_v_' + str(s_v_var) + '_intens_a_' + str(s_a_intensity) + '_intens_v_' + str(s_v_intensity) + '_draws_' + str(n_draws)


# exp_dir = path = os.path.join(os.getcwd(), 'Results')

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
    with open(os.path.join(exp_dir, exp_name_neurons + '.txt',), 'w+') as f:  # Python 3: open(..., 'wb')
        f.write(exp_name + '\n \n')
        f.write('Audio Stimulus Mean  : ' + str(s_a_mean) + '\n')
        f.write('Audio Stimulus Variance  : ' + str(s_a_var) + '\n')
        f.write('Audio Stimulus Intensity  : ' + str(s_a_intensity) + '\n')
        f.write('Visual Stimulus Mean  : ' + str(s_v_mean) + '\n')
        f.write('Visual Stimulus Variance : ' + str(s_v_var) + '\n')
        f.write('Visual Stimulus Intensity : ' + str(s_v_intensity) + '\n')
        f.write('\n \n')
        f.write('Sensory Audio Sigma  : ' + str(sigma_s_a) + '\n')
        f.write('Sensory Video Sigma  : ' + str(sigma_s_v) + '\n')
        f.write('Cortical Audio Sigma : ' + str(sigma_c_a) + '\n')
        f.write('Cortical Video Sigma : ' + str(sigma_c_v) + '\n')
        f.write('\n \n')
        f.write('Conditions to test : ' + str(conditions_to_test) + '\n')
        f.write('\n \n')
        f.close()


# %%
########################################
# initiualize arrays

net_out = np.zeros((n_draws, len(conditions_to_test), net.n_neurons_msi))
sensory_input_v = np.zeros(
    (n_draws, len(conditions_to_test), net.len_t, net.n_neurons_msi))
sensory_input_a = np.zeros(
    (n_draws, len(conditions_to_test), net.len_t, net.n_neurons_msi))
cortical_input_v = np.zeros(
    (n_draws, len(conditions_to_test), net.len_t, net.n_neurons_msi))
cortical_input_a = np.zeros(
    (n_draws, len(conditions_to_test), net.len_t, net.n_neurons_msi))
r_all = np.zeros((n_draws, len(conditions_to_test),
                  net.len_t, net.n_neurons_msi))
p_pool_all = np.zeros((n_draws, len(conditions_to_test),
                       net.len_t, net.n_neurons_msi))
p_sensory_all = np.zeros(
    (n_draws, len(conditions_to_test), net.len_t, net.n_neurons_msi))
q_fb_all = np.zeros((n_draws, len(conditions_to_test),
                     net.len_t, net.n_neurons_msi))
q_s2_v_all = np.zeros((n_draws, len(conditions_to_test),
                       net.len_t, net.n_neurons_msi))
q_s2_a_all = np.zeros((n_draws, len(conditions_to_test),
                       net.len_t, net.n_neurons_msi))
q_s1_v_all = np.zeros((n_draws, len(conditions_to_test),
                       net.len_t, net.n_neurons_msi))
q_s1_a_all = np.zeros((n_draws, len(conditions_to_test),
                       net.len_t, net.n_neurons_msi))


if not skip_simulation:
    for i_draw in range(n_draws):

        s_a_location = -1
        s_v_location = -1

        while s_a_location < 0 or s_a_location >= n_neurons_msi:
            s_a_location = np.random.normal(loc=s_a_mean, scale=s_a_var)

        # draw stimulus
        while s_v_location < 0 or s_v_location >= n_neurons_msi:
            s_v_location = np.random.normal(loc=s_v_mean, scale=s_v_var)

        # draw stimulus location and intensity from distributions
        for i_condi, condi in enumerate(conditions_to_test):

            ########################
            # Create Input Stimuli #
            ########################

            # sensory inputs
            stimuli_s_v = {'loc': np.array([s_v_location]), 'onset': s_onset_temp,
                           'duration': net.len_t - s_onset_temp, 'sigma': sigma_s_v, 'weight': s_v_intensity}
            stimuli_s_a = {'loc': np.array([s_a_location]), 'onset': s_onset_temp,
                           'duration': net.len_t - s_onset_temp, 'sigma': sigma_s_a, 'weight': s_a_intensity}

            # cortical inputs
            stimuli_c_v = {'loc': np.array([s_v_location]), 'onset': s_onset_temp,
                           'duration': net.len_t - s_onset_temp, 'sigma': sigma_c_v, 'weight': s_v_intensity}
            stimuli_c_a = {'loc': np.array([s_a_location]), 'onset': s_onset_temp,
                           'duration': net.len_t - s_onset_temp, 'sigma': sigma_c_a, 'weight': s_a_intensity}

            # Create inputs
            sens_in_v, sens_in_a, cor_in_v, cor_in_a = net.create_inputs(
                stimuli_s_v, stimuli_s_a, stimuli_c_v, stimuli_c_a, gauss=True)

            # run the network with random locations
            r, act, p_pool, p_sensory, q_fb, q_s2_v, q_s2_a, q_s1_v, q_s1_a = net.run(
                i_condi, dirac_kernels=no_spatial_conv)

            # save the data
            net_out[i_draw, i_condi, :] = act[readout_time, :]
            r_all[i_draw, i_condi, :, :] = r
            # p_pool_all[i_draw, i_condi, :, :] = p_pool
            # p_sensory_all[i_draw, i_condi, :, :] = p_sensory
            q_fb_all[i_draw, i_condi, :, :] = q_fb
            # q_s2_v_all[i_draw, i_condi, :, :] = q_s2_v
            # q_s2_a_all[i_draw, i_condi, :, :] = q_s2_a
            # q_s1_v_all[i_draw, i_condi, :, :] = q_s1_v
            # q_s1_a_all[i_draw, i_condi, :, :] = q_s1_a
            sensory_input_v[i_draw, i_condi, :, :] = sens_in_v
            sensory_input_a[i_draw, i_condi, :, :] = sens_in_a
            # cortical_input_v[i_draw, i_condi, :, :] = cor_in_v
            # cortical_input_a[i_draw, i_condi, :, :] = cor_in_a

            print('Draw ' + str(i_draw + 1) + ' of ' + str(n_draws) + ' Condition : ' +
                  str(i_condi + 1) + ' of ' + str(len(conditions_to_test)))

# %%
###### Save outputs ######

results_file = os.path.join(exp_dir, 'results.pkl')
if not os.path.exists(results_file):
    with open(results_file, 'wb') as f:
        pickle.dump([net_out, r_all, sensory_input_v, sensory_input_a], f)
else:
    with open(results_file, 'rb') as f:
        net_out, r_all, sensory_input_v, sensory_input_a = pickle.load(
            f)


###### Plotting ######
# %%

fusion = np.zeros(n_draws).astype('bool')
for i_draw in range(n_draws):
    fusion[i_draw] = not (
        len(find_peaks(np.squeeze(net_out[i_draw, 1, :]), distance=1)[0]) > 1)


# find all modes of response
modes_response_fb_on = np.argmax(net_out[fusion, 1, :], 1)
modes_response_fb_off = np.argmax(net_out[fusion, 0, :], 1)

# find all modes of inputs
modes_input_a = np.argmax(sensory_input_a[fusion, 1, readout_time, :], 1)
modes_input_v = np.argmax(sensory_input_v[fusion, 1, readout_time, :], 1)

fig = plt.figure(figsize=(10, 10))
# plot the stuff
plt.hist(modes_response_fb_on, bins=21, range=(0, 20), alpha=0.5)
plt.hist(modes_response_fb_off, bins=21, range=(
    0, 20), histtype='step', linestyle=('dashed'))
plt.hist(modes_input_a, bins=21, range=(0, 20), histtype='step')
plt.hist(modes_input_v, bins=21, range=(0, 20), histtype='step')

# caluclate means and vars from response
res_mean_fb_off = np.argmax(np.histogram(
    modes_response_fb_off, bins=21, range=(0, 20))[0])
res_mean_fb_on = np.argmax(np.histogram(
    modes_response_fb_on, bins=21, range=(0, 20))[0])
res_var_fb_off = np.var(modes_response_fb_off)
res_var_fb_on = np.var(modes_response_fb_on)
sens_a_var = np.var(modes_input_a)
sens_v_var = np.var(modes_input_v)
# calculate means and vars from input
computed_mean = np.argmax(np.mean(
    sensory_input_a[fusion, 1, readout_time, :] * sensory_input_v[fusion, 1, readout_time, :], 0))
computed_var = (sens_a_var * sens_v_var) / (sens_a_var + sens_v_var)

print('\nModel Response Mean (Cort On): {0:.2f} \nModel Response Mean (Cort Off): {1:.2f} \nComputed Mean   : {2:.2f}'.format(
    res_mean_fb_on, res_mean_fb_off, computed_mean))
print('\nModel Response Variance (Cort On): {0:.2f} \nModel Response Variance (Cort Off): {1:.2f} \nComputed Variance   : {2:.2f}'.format(
    res_var_fb_on, res_var_fb_off, computed_var))

# save stuff
results_file = os.path.join(exp_dir, 'means_vars.pkl')
with open(results_file, 'wb') as f:
    pickle.dump([res_mean_fb_on, res_var_fb_on, computed_mean,
                 computed_var, res_mean_fb_off, res_var_fb_off], f)


# %%

q_fb_all[:, :, 3950, 8]
