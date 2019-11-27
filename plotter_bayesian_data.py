import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


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

    def calculate_mean_std(self, x, y):
        """ Computes the mean and standard devitaion from scatter plot values """
        # combine the values
        tmp = np.array([x, y])
        # get unique valvues
        u = np.unique(tmp[0, :])

        # sort the y values accoring to x values (this is a left over but it works)
        data_frame = np.zeros((len(u), tmp.shape[1]))
        data_frame[:] = np.nan
        for i_unique, e_unique in enumerate(u):
            a = np.where(e_unique == tmp)[1]
            data_frame[i_unique, a] = tmp[1, a]

        means = np.nanmean(data_frame, 1)
        stds = np.nanstd(data_frame, 1)

        return means, stds

    def add_mean_std_plt(self, ax, x, y, color='C1'):
        means, stds = self.calculate_mean_std(x, y)
        ax.plot(np.unique(x), means, color=color)
        ax.fill_between(np.unique(x), means -
                        stds, means + stds, color=color, alpha=0.25)
        return ax

    def plot_mean_var_comparison(self, means, vars):
        fig = plt.figure(figsize=self.fig_size)

        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('Mean Comparison')
        ax.scatter(means[:, -1], means[:, 0],
                   label='Cortical FB ON', facecolors='C1', edgecolors='C1')
        ax.scatter(means[:, -1], means[:, 1], facecolors='none',
                   edgecolors='black', label='Cortical FB OFF')
        ax.plot([0, 20], [0, 20], 'black', linestyle='--')
        ax.set_xlabel('Computed Mean')
        ax.set_ylabel('Model Mean')

        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Variance Comparison')
        ax.scatter(vars[:, -1], vars[:, 0], label='Cortical FB ON',
                   facecolors='C1', edgecolors='C1')
        ax.scatter(vars[:, -1], vars[:, 1], facecolors='none',
                   edgecolors='black', label='Cortical FB OFF')
        ax.plot([0, 4], [0, 4], 'black', linestyle='--')
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 4])
        ax.set_xlabel('Computed Variance')
        ax.set_ylabel('Model Variance')

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path,
                                     'Fusion_Prob.' + self.format), dpi=300)
        plt.legend()

        return ax

    def plot_intens_var_comparison(self, x, vars, lin_reg=False):
        fig = plt.figure(figsize=self.fig_size)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Intensity Influence')
        ax.scatter(x[:, 0], np.abs(vars[:, -1] - vars[:, 0]),
                   label='Cortical FB ON', facecolors='C1', edgecolors='C1')
        ax.scatter(x[:, 0], np.abs(vars[:, -1] -
                                   vars[:, 1]), facecolors='none', edgecolors='black', label='Cortical FB OFF')
        ax.set_xlabel('Intensities')
        ax.set_ylabel('Abs Variance Error (Computed - Model)')
        ax.set_ylim([0, 4])

        if lin_reg:
            lr = LinearReg(x[:, 0].flatten(), (np.abs(
                vars[:, -1] - vars[:, 0])).flatten())
            x_r, y_r = lr.get_fitted_line()
            ax.plot(np.squeeze(x_r), np.squeeze(y_r), color='C1', linewidth=1)

            lr = LinearReg(x[:, 0].flatten(), (np.abs(
                vars[:, -1] - vars[:, 1])).flatten())
            x_r, y_r = lr.get_fitted_line()
            ax.plot(np.squeeze(x_r), np.squeeze(
                y_r), color='black', linewidth=1)

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path,
                                     'Fusion_Prob.' + self.format), dpi=300)
        plt.legend()

        return ax

    def plot_location_var_comparison(self, x, vars,lin_reg=False):
        fig = plt.figure(figsize=self.fig_size)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Spatial Offset Influence')
        ax.scatter(x[:, 0] - x[:, 1],
                   np.abs(vars[:, -1] - vars[:, 0]), label='Cortical FB ON', facecolors='C1', edgecolors='C1')
        ax.scatter(x[:, 0] - x[:, 1],
                   np.abs(vars[:, -1] - vars[:, 1]), facecolors='none', edgecolors='black', label='Cortical FB OFF')
        ax.set_xlabel('Spatial Offset')
        ax.set_ylabel('Abs Variance Error (Computed - Model)')
        ax.set_ylim([0, 4])

        if lin_reg:
            lr = LinearReg((x[:, 0] - x[:, 1]).flatten(), (np.abs(
                vars[:, -1] - vars[:, 0])).flatten())
            x_r, y_r = lr.get_fitted_line()
            ax.plot(np.squeeze(x_r), np.squeeze(y_r), color='C1', linewidth=1)

            lr = LinearReg((x[:, 0] - x[:, 1]).flatten(), (np.abs(
                vars[:, -1] - vars[:, 1])).flatten())
            x_r, y_r = lr.get_fitted_line()
            ax.plot(np.squeeze(x_r), np.squeeze(
                y_r), color='black', linewidth=1)

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path,
                                     'Fusion_Prob.' + self.format), dpi=300)
        plt.legend()

        return ax

    def plot_var_var_comparison(self, x, vars, lin_reg=False):
        fig = plt.figure(figsize=self.fig_size)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Variance Offset Influence')
        ax.scatter(x[:, 0] - x[:, 1],
                   np.abs(vars[:, -1] - vars[:, 0]), label='Cortical FB ON', facecolors='C1', edgecolors='C1')
        ax.scatter(x[:, 0] - x[:, 1],
                   np.abs(vars[:, -1] - vars[:, 1]), facecolors='none', edgecolors='black', label='Cortical FB OFF')
        ax.set_xlabel('Variance Offset (Var A - Var V)')
        ax.set_ylabel('Abs Variance Error (Computed - Model)')
        ax.set_ylim([0, 4])

        if lin_reg:
            lr = LinearReg((x[:, 0] - x[:, 1]).flatten(), (np.abs(
                vars[:, -1] - vars[:, 0])).flatten())
            x_r, y_r = lr.get_fitted_line()
            ax.plot(np.squeeze(x_r), np.squeeze(y_r), color='C1', linewidth=1)

            lr = LinearReg((x[:, 0] - x[:, 1]).flatten(), (np.abs(
                vars[:, -1] - vars[:, 1])).flatten())
            x_r, y_r = lr.get_fitted_line()
            ax.plot(np.squeeze(x_r), np.squeeze(
                y_r), color='black', linewidth=1)

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path,
                                     'Fusion_Prob.' + self.format), dpi=300)
        plt.legend()

        return ax

    def plot_fusion_prob(self, spatial_offsets, fusion_fb_on_percentage, fusion_fb_off_percentage='None', std_plot=True):
        fig = plt.figure(figsize=self.fig_size)

        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Probability of Event Fusion')

        ax.scatter(spatial_offsets, fusion_fb_on_percentage,
                   facecolors='C1', edgecolors='C1', label='Cortical FB ON')

        if std_plot:
            ax = self.add_mean_std_plt(ax,
                                       spatial_offsets, fusion_fb_on_percentage, color='C1')

        if np.any(fusion_fb_off_percentage):
            ax.scatter(spatial_offsets, fusion_fb_off_percentage,
                       facecolors='none', edgecolors='black', label='Cortical FB OFF')
            if std_plot:
                ax = self.add_mean_std_plt(ax,
                                           spatial_offsets, fusion_fb_off_percentage, color='black')

        ax.set_xlabel('Spatial Offset')
        ax.set_ylabel('P(single Event)')

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path,
                                     'Fusion_Prob.' + self.format), dpi=300)
        plt.legend()

        return ax

    def plot_fitted_sigmoid(self, spatial_offsets, fusion_fb_on_percentage, fusion_fb_off_percentage='None'):

        # define the lambda function
        def fsigmoid(x, a, b): return 1.0 / (1.0 + np.exp(-a * (x - b)))

        fig = plt.figure(figsize=self.fig_size)

        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Probability of Event Fusion')

        means, stds = self.calculate_mean_std(
            spatial_offsets, fusion_fb_on_percentage)

        popt, pcov = curve_fit(fsigmoid, np.unique(
            spatial_offsets), means[::-1] / 100, method='dogbox', bounds=([0., 0.], [10, 10]))

        ax.scatter(np.unique(spatial_offsets), means[::-1] / 100, color='C1')
        ax.plot(np.linspace(0, 9, 100), fsigmoid(
            np.linspace(0, 9, 100), *popt), color='C1', label='Cortical FB ON')

        if np.any(fusion_fb_off_percentage):

            means, stds = self.calculate_mean_std(
                spatial_offsets, fusion_fb_off_percentage)

            popt, pcov = curve_fit(fsigmoid, np.unique(
                spatial_offsets), means[::-1] / 100, method='dogbox', bounds=([0., 0.], [10, 10]))

            ax.scatter(np.unique(spatial_offsets),
                       means[::-1] / 100, color='black')
            ax.plot(np.linspace(0, 9, 100), fsigmoid(
                np.linspace(0, 9, 100), *popt), color='black', label='Cortical FB OFF')

        ax.invert_xaxis()
        ax.set_xticklabels(ax.get_xticklabels()[::-1])

        ax.set_xlabel('Spatial Offset')
        ax.set_ylabel('P(single Event)')

        if self.save_figs:
            plt.savefig(os.path.join(self.save_path,
                                     'Fusion_Prob.' + self.format), dpi=300)
        plt.legend()

        return ax


class LinearReg():

    def __init__(self, x, y):
        from sklearn.linear_model import LinearRegression

        self.lr_model = LinearRegression()

        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        self.lr_model.fit(self.x, self.y)

        self.rr = self.lr_model.score(self.x, self.y)

    def get_fitted_line(self):
        return [self.x, self.lr_model.predict(self.x)]

    def get_coefficients(self):
        return self.lr_model.coef_[0, 0], self.lr_model.intercept_[0]

    def get_score(self, x=0, y=0):
        if x == 0 or y == 0:
            return self.rr
        else:
            return self.lr_model.score(x, y)

    def print_coefficients(self):
        print('Gain: {0:1.2f}, Bias: {1:1.2f}, , r^2: {2:1.2f}'.format(
            self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr))
        return ('Gain: {0:1.2f},\nBias: {1:1.2f},\n' + r'$r^2$: {2:1.2f}').format(self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr)
