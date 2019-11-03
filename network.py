import numpy as np
#import matplotlib.pyplot as plt


class Network(object):
    """docstring for Network."""

    def __init__(self, experiment_name, simulation_time=4, n_neurons_msi=20):

        ###### Variables ######

        # Arguments
        self.experiment_name = experiment_name
        self.simulation_time = simulation_time
        self.n_neurons_msi = n_neurons_msi

        # Internal Variables

        # Neuron Parameters

        # helper variables
        self.dt = 0.001

        # Initialize Variables
        self.initialize_simulation_vars()

    def initialize_simulation_vars(self):
        # simulation time
        self.t = np.linspace(0, self.simulation_time, int(
            self.simulation_time / self.dt) + 1)
        self.len_t = len(self.t)
        # state var r
        self.r = np.zeros((self.len_t, self.n_neurons_msi))

    def create_inputs(self, stimuli_s_v, stimuli_s_a, stimuli_c_v, stimuli_c_a):

        sensory_input_v = np.zeros((self.len_t, self.n_neurons_msi))
        sensory_input_a = np.zeros((self.len_t, self.n_neurons_msi))
        cortical_input_v = np.zeros((self.len_t, self.n_neurons_msi))
        cortical_input_a = np.zeros((self.len_t, self.n_neurons_msi))
        # Sensory Inputs
        location = stimuli_s_v['loc']
        onset = stimuli_s_v['onset']
        duration = stimuli_s_v['duration']
        sigma = stimuli_s_v['sigma']
        weight = stimuli_s_v['weight']
        x = np.arange(0, self.n_neurons_msi)

        sensory_input_v[onset:onset + duration,
                        :] = gauss_normalized(x, location[0], sigma=sigma) * weight
        # normalize input
#             sensory_input_v /= np.max(sensory_input_v)

        location = stimuli_s_a['loc']
        onset = stimuli_s_a['onset']
        duration = stimuli_s_a['duration']
        sigma = stimuli_s_a['sigma']
        weight = stimuli_s_a['weight']

        sensory_input_a[onset:onset + duration,
                        :] = gauss_normalized(x, location[0], sigma=sigma) * weight
#             sensory_input_a /= np.max(sensory_input_a)

        location = stimuli_c_v['loc']
        onset = stimuli_c_v['onset']
        duration = stimuli_c_v['duration']
        sigma = stimuli_c_v['sigma']
        weight = stimuli_c_v['weight']

        cortical_input_v[onset:onset + duration,
                         :] = gauss_normalized(x, location[0], sigma=sigma) * weight
#             cortical_input_v /= np.max(cortical_input_v)

        # Cortical Inputs
        location = stimuli_c_a['loc']
        onset = stimuli_c_a['onset']
        duration = stimuli_c_a['duration']
        sigma = stimuli_c_a['sigma']
        weight = stimuli_c_a['weight']

        cortical_input_a[onset:onset + duration,
                         :] = gauss_normalized(x, location[0], sigma=sigma) * weight
#             cortical_input_a /= np.max(cortical_input_a)

        assert sensory_input_v.shape == (
            self.len_t, self.n_neurons_msi), 'Input Dimension Mismatch'
        assert sensory_input_a.shape == (
            self.len_t, self.n_neurons_msi), 'Input Dimension Mismatch'
        assert cortical_input_v.shape == (
            self.len_t, self.n_neurons_msi), 'Input Dimension Mismatch'
        assert cortical_input_a.shape == (
            self.len_t, self.n_neurons_msi), 'Input Dimension Mismatch'

        self.sensory_input_v = sensory_input_v
        self.sensory_input_a = sensory_input_a
        self.cortical_input_v = cortical_input_v
        self.cortical_input_a = cortical_input_a

        return sensory_input_v, sensory_input_a, cortical_input_v, cortical_input_a

    def run(self, condition_number):

        # store the output
        r = np.zeros((self.len_t, self.n_neurons_msi))
        p_sensory = np.zeros((self.len_t, self.n_neurons_msi))
        p_pool = np.zeros((self.len_t, self.n_neurons_msi))
        q_fb = np.zeros((self.len_t, self.n_neurons_msi))
        q_s2_v = np.zeros((self.len_t, self.n_neurons_msi))
        q_s2_a = np.zeros((self.len_t, self.n_neurons_msi))
        q_s1_a = np.zeros((self.len_t, self.n_neurons_msi))
        q_s1_v = np.zeros((self.len_t, self.n_neurons_msi))

        sensory_input_v = self.sensory_input_v
        sensory_input_a = self.sensory_input_a
        cortical_input_v = self.cortical_input_v
        cortical_input_a = self.cortical_input_a

        if condition_number == 0:
            # cortical projections off
            cortical_fb = False
            cortical_input_a[:, :] = 0
            cortical_input_v[:, :] = 0
        elif condition_number == 1:
            cortical_fb = True
        elif condition_number == 2:
            cortical_fb = True
            cortical_input_v[:, :] = 0
        elif condition_number == 3:
            cortical_fb = True
            cortical_input_a[:, :] = 0
        elif condition_number == 4:
            cortical_fb = True
            sensory_input_a[:, :] = 0
            cortical_input_a[:, :] = 0
        elif condition_number == 5:
            cortical_fb = True
            sensory_input_v[:, :] = 0
            cortical_input_v[:, :] = 0
        elif condition_number == 6:
            cortical_fb = True
            sensory_input_v[:, :] = 0

        # run the network for simulation time
        for t in range(0, self.len_t - 1):

            # walk over all neurons in the map (map loop)
            x_kernel = np.arange(self.n_neurons_msi)
            for i_neurons in range(self.n_neurons_msi):

                kernel = gauss_normalized(x_kernel, i_neurons, 1.0)
                kernel_mod_fb = gauss_normalized(x_kernel, i_neurons, 3.0)
                kernel_pool = gauss_normalized(x_kernel, i_neurons, 1.0)

                # calculate the sensory inter neuron inputs and acti_neuronsation
                excitatory_in = sensory_input_a[t, i_neurons] * \
                    sensory_input_v[t, i_neurons] * 200
                p_sensory[t + 1, i_neurons] = p_sensory[t, i_neurons] + \
                    self.dt * \
                    ode_p_sensory(p_sensory[t, i_neurons], excitatory_in)

                # calculate the ms neuron inputs and acti_neuronsation
                excitatory_in = sensory_input_a[t,
                                                i_neurons] + sensory_input_v[t, i_neurons]
                # pool exists only once
                inhibitory_in = np.sum(kernel * out_pool_thres(p_pool[t, :], 0.0, 1)) + np.sum(
                    kernel * out_thres(p_sensory[t, :], slope=2))  # + out_thres(q_sensory_v[  t,i_neurons  ])

                if cortical_fb:
                    modulatory_in = np.sum(
                        kernel_mod_fb * out_thres(q_fb[t, :]))
                else:
                    modulatory_in = 0

                r[t + 1, i_neurons, ] = r[t, i_neurons] + self.dt * \
                    ode_r_msi(r[t, i_neurons], excitatory_in,
                              inhibitory_in, modulatory_in)

                # calculate the pool neuron inputs and acti_neuronsation
                # acti_neuronsity depends on previous r acitvity
                excitatory_in = np.sum(
                    kernel_pool * out_sigmoid(r[t, :], slope=1.7))
                p_pool[t + 1, i_neurons] = p_pool[t, i_neurons] + \
                    self.dt * ode_p_pool(p_pool[t, i_neurons], excitatory_in)

                # calculate the feedback neurons inputs and acti_neuronsation
                excitatory_in = cortical_input_a[t,
                                                 i_neurons] + cortical_input_v[t, i_neurons]
                inhibitory_in = np.sum(kernel * out_thres(q_s2_v[t, :], slope=1)) + np.sum(
                    kernel * out_thres(q_s2_a[t, :], slope=1))
                q_fb[t + 1, i_neurons] = q_fb[t, i_neurons, ] + self.dt * \
                    ode_q_fb(q_fb[t, i_neurons], excitatory_in, inhibitory_in)

                # calculate the cortical neuron stage 2 V inputs and acti_neuronsation
                excitatory_in = cortical_input_v[t, i_neurons]
                inhibitory_in = np.sum(
                    kernel * out_thres(q_s1_v[t, i_neurons], slope=1))
                q_s2_v[t + 1, i_neurons] = q_s2_v[t, i_neurons] + self.dt * \
                    ode_q_s2(q_s2_v[t, i_neurons],
                             excitatory_in, inhibitory_in)

                # calculate the cortical neuron stage 2 A inputs and acti_neuronsation
                excitatory_in = cortical_input_a[t, i_neurons]
                inhibitory_in = np.sum(
                    kernel * out_thres(q_s1_a[t, i_neurons], slope=1))
                q_s2_a[t + 1, i_neurons] = q_s2_a[t, i_neurons] + self.dt * \
                    ode_q_s2(q_s2_a[t, i_neurons],
                             excitatory_in, inhibitory_in)

                # calculate the cortical neuron stage 1 V inputs and acti_neuronsation
                excitatory_in = cortical_input_a[t, i_neurons]
                q_s1_v[t + 1, i_neurons] = q_s1_v[t, i_neurons] + \
                    self.dt * ode_q_s1(q_s1_v[t, i_neurons], excitatory_in)

                # calculate the cortical neuron stage 1 A inputs and acti_neuronsation
                excitatory_in = cortical_input_v[t, i_neurons]
                q_s1_a[t + 1, i_neurons] = q_s1_a[t, i_neurons] + \
                    self.dt * ode_q_s1(q_s1_a[t, i_neurons], excitatory_in)

        return r, out_sigmoid(r, slope=1.7), p_pool, p_sensory, q_fb, q_s2_v, q_s2_a, q_s1_v, q_s1_a


############################################################
#                          ODEs                            #
############################################################
# define the ODE for the SC multisensory neuron
def ode_r_msi(r, excitatory_in=0, inhibitory_in=0, modulatory_in=0):
    # tau defines how fast the membrane potential builds up
    tau = 1
    # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
    alpha = 1
    # beta defines the upper limit of the membrane potential
    beta = 1
    # gamma defines the subtracti_neuronse influence of the inhibitory input
    gamma = 0.0
    # kappa defines the di_neuronsisi_neuronse influence of the inhibitory input
    kappa = 0.25
    # defines the influence of the modulatory input
    lambdaa = 0.4

    # calculate the change of r_Alearn
    d_r = -alpha * r + (beta - r) * excitatory_in * (1 + lambdaa *
                                                     modulatory_in) - (gamma + kappa * r) * inhibitory_in

    return d_r / tau

    # define the ODE for the sensory interneuron neuron


def ode_p_sensory(p, excitatory_in=0):
 # tau defines how fast the membrane potential builds up
    tau = 1
    # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
    alpha = 1
    # beta defines the upper limit of the membrane potential
    beta = 1

    # calculate the change of r_Alearn
    d_p = -alpha * p + (beta - p) * excitatory_in

    return d_p / tau

# define the ODE for the pool inhibitory neuron


def ode_p_pool(p, excitatory_in=0):
 # tau defines how fast the membrane potential builds up
    tau = 1  # slower than normal neurons
    # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
    alpha = 1.0
    # beta defines the upper limit of the membrane potential
    beta = 1.0

    # calculate the change of r_Alearn
    d_p = -alpha * p + (beta - p) * excitatory_in

    return d_p / tau

# define the ODE for the cortical feedback neuron


def ode_q_fb(q, excitatory_in=0, inhibitory_in=0):
    # tau defines how fast the membrane potential builds up
    tau = 1
    # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
    alpha = 1.0
    # beta defines the upper limit of the membrane potential
    beta = 2
    # gamma defines the subtracti_neuronse influence of the inhibitory input
    gamma = 5.0
    # kappa defines the di_neuronsisi_neuronse influence of the inhibitory input
    kappa = 1

    # calculate the change of r_Alearn
    d_q = -alpha * q + (beta - q) * excitatory_in - \
        (gamma + kappa * q) * inhibitory_in

    return d_q / tau

# define the ODE for the cortical inhibitory neuron on stage 2


def ode_q_s2(q, excitatory_in=0, inhibitory_in=0):
    # tau defines how fast the membrane potential builds up
    tau = 1
    # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
    alpha = 1.0
    # beta defines the upper limit of the membrane potential
    beta = 1
    # gamma defines the subtracti_neuronse influence of the inhibitory input
    gamma = 5.0
    # kappa defines the di_neuronsisi_neuronse influence of the inhibitory input
    kappa = 1

    # calculate the change of r_Alearn
    d_q = -alpha * q + (beta - q) * excitatory_in - \
        (gamma + kappa * q) * inhibitory_in

    return d_q / tau


# define the ODE for the cortical inhibitory neuron on stage 1
def ode_q_s1(q, excitatory_in=0):
 # tau defines how fast the membrane potential builds up
    tau = 1.0  # slower than normal neurons
    # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
    alpha = 1.0
    # beta defines the upper limit of the membrane potential
    beta = 1.0

    # calculate the change of r_Alearn
    d_q = -alpha * q + (beta - q) * excitatory_in

    return d_q / tau

############################################################
#                 Activation Functions                     #
############################################################

# Defines the output transfer function of q_A_SC


def out_thres(x, threshold=0.0, slope=1):
    return np.minimum(np.maximum((x - threshold) * slope, 0), 1)

# Defines the output transfer function of q_A_SC


def out_pool_thres(x, threshold=0.0, slope=1):
    return np.minimum(np.maximum((x - threshold) * slope, 0), 1)

# Defines the sigmoidal transfer function


def out_sigmoid(x, slope=5, operation_point=0.0):
    tmp = (x - operation_point) * slope
    return (2 / (1 + np.exp(-4 * tmp**2))) - 1


############################################################
#                   Helpfer Functions                      #
############################################################


# define a gauss probability function
def gauss_prob_distr(x, mean, sigma):

    if sigma == 0.0:
        return np.zeros(x.shape)
    x
    l = len(x)
    x = np.arange(0, l * 3)

    x_s = mean + l
    tmp = np.exp(-((x - x_s) ** 2) / (2 * sigma ** 2))
    tmp /= np.sum(tmp)

    tmp = tmp[l:-l]
    return tmp

# define a gauss that is max normalized


def gauss_normalized(x, mean, sigma):

    if sigma == 0.0:
        return np.zeros(x.shape)
    else:
        tmp = np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
        return tmp / np.max(tmp)
