import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class Plotter(object):
    """docstring for Plotter."""

    def __init__(self, save_path='./', save_figs=False, fig_size=(10, 10), drawing_size=20, format='pdf'):
        """docstring for Plotter."""
        super(Plotter, self).__init__()

        self.save_figs = save_figs
        self.fig_size = fig_size
        self.format = format
        self.save_path = save_path

        plt.style.use('seaborn-whitegrid')

        mpl.rcParams['grid.linestyle'] = ':'

        mpl.rcParams['figure.figsize'] = [10.0, 10.0]
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 400

        mpl.rcParams['font.size'] = drawing_size
        # mpl.rcParams['font.style'] = 'oblique'
        mpl.rcParams['font.weight'] = 'heavy'
        mpl.rcParams['font.family'] = ['DejaVu Sans']

        mpl.rcParams['figure.titlesize'] = int(drawing_size * 1.3)
        mpl.rcParams['figure.titleweight'] = 'heavy'

        mpl.rcParams['lines.linewidth'] = int(drawing_size / 5)

        mpl.rcParams['axes.labelsize'] = drawing_size
        mpl.rcParams['axes.labelweight'] = 'heavy'
        mpl.rcParams['axes.titlesize'] = int(drawing_size * 1.3)
        mpl.rcParams['axes.titleweight'] = 'heavy'

        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['legend.fontsize'] = int(drawing_size * 0.9)
        mpl.rcParams['legend.frameon'] = True
        mpl.rcParams['legend.framealpha'] = 0.5
        mpl.rcParams['legend.facecolor'] = 'inherit'
        mpl.rcParams['legend.edgecolor'] = '0.8'

    def plot_normal_response(self, net_out, x_location, sensory_input_v, sensory_input_a, spatial_offsets, sensory_intensities, conditions_to_test):

        fig = plt.figure(figsize=self.fig_size)

        ax1r, ax2r, ax3r = fig.subplots(
            3, len(spatial_offsets), sharex=False, sharey='row', squeeze=False)

        for i_offset in range(len(spatial_offsets)):

            ax = ax1r[int(i_offset)]

            ax.plot(np.arange(net_out.shape[3]), np.squeeze(
                sensory_input_v[i_offset, :]), color='C0', marker='o', label='Visual Input')
            ax.plot(np.arange(net_out.shape[3]), np.squeeze(
                sensory_input_a[i_offset, :]), color='C3', marker='o', label='Audio Input')
            ax.set_ylim([0, 1])

            # ax.set_title('Offset: '+str(np.abs(stimuli_s_v[i]['loc'][0]-stimuli_s_a[i]['loc'][0])))
            if i_offset == 0:
                ax.set_ylabel('Input Stimulus')

            ax = ax2r[int(i_offset)]

            # combined response cortical feedback off
            if 0 in conditions_to_test:
                # combined response cortical feedback off
                cr_all_off = net_out[i_offset, :, 0, x_location]
                ax.plot(sensory_intensities, cr_all_off, color='black',
                        marker='o', label='Cortical Feedback OFF')

            if 1 in conditions_to_test:
                # combined response cortical feedback on
                cr_all_on = net_out[i_offset, :, 1, x_location]
                ax.plot(sensory_intensities, cr_all_on, color='C1',
                        marker='o', label='Cortical Feedback ON')
                # #combined response visual cortical feedback off
            if 2 in conditions_to_test:
                cr_vis_off = net_out[i_offset, :, 2, x_location]
                ax.plot(sensory_intensities, cr_vis_off, color='C6',
                        marker='o', label='Bimodal Stimuli, Visual Feedback OFF')
            if 3 in conditions_to_test:
                # #combined response auditory cortical feedback off
                cr_aud_off = net_out[i_offset, :, 3, x_location]
                ax.plot(sensory_intensities, cr_aud_off, color='C4',
                        marker='o', label='Bimodal Stimuli, Audio Feedback OFF')
            if 4 in conditions_to_test:
                # #single visual response visual cortical feedback off
                vi_vis_off = net_out[i_offset, :, 4, x_location]
                ax.plot(sensory_intensities, vi_vis_off, color='C0',
                        marker='o', label='Visual Only, Sensory + Feedback')
            if 5 in conditions_to_test:
                # single auditory response auditory cortical feedback off
                au_aud_off = net_out[i_offset, :, 5, x_location]
                ax.plot(sensory_intensities, au_aud_off, color='C3',
                        marker='o', label='Audio Only, Sensory + Feedback')

            if 4 in conditions_to_test and 5 in conditions_to_test:
                ax.plot(sensory_intensities, au_aud_off + vi_vis_off,
                        color='black', linestyle='--', label='SUM')

            # #single auditory response cortical feedback on
        #    au_all_off =  out_sigmoid(r[:,6,probing_times,x_location,y_location,i],slope )
        #    ax.plot(sensory_intensities, au_all_off ,color= 'C5',marker='o', linestyle='-' ,label='Auditory Stimuli, (Bimodal) Feedback ON')

            #
            if i_offset == 0:
                ax.set_ylabel('Activity')
            ax.set_ylim([0, 1.5])

            if 4 in conditions_to_test and 5 in conditions_to_test:

                # activity index
                ax1 = ax3r[int(i_offset)]
                # this index cannot be calculated for input intensity 0
                intensity_range = np.arange(1, len(sensory_intensities))
                activity_index = cr_all_on[intensity_range] / \
                    (vi_vis_off[intensity_range] + au_aud_off[intensity_range])

                ax1.plot(sensory_intensities[intensity_range],
                         activity_index, marker='o', color='C1')
                ax1.hlines(1.0, 0, 1, linestyle='--')
                ax1.set_ylim([0, 2.0])

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path, 'Normal_response.' + self.format), dpi=300)

        plt.show()
        return ax
