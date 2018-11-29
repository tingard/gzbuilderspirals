import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from . import metric, clustering, cleaning, deprojecting, fitting, pipeline
from . import r_theta_from_xy, get_sample_weight, equalize_arm_length, _vprint


_imshow_kwargs = dict(origin='lower', cmap='gray_r')


def _figure_wrapper(f):
    def _func(*args, outfile=None, figsize=(8, 8), **kwargs):
        plt.figure(figsize=figsize)
        kw = f(*args, **kwargs)
        if outfile is not None:
            save_kwargs = kw if kw is not None else {}
            plt.savefig(outfile, **save_kwargs)
        plt.close()
    return _func


class FigureWrapper():
    def __init__(self, func):
        self.func = func
        self.sets = []

    def add(self, *args, **kwargs):
        self.sets.append([args, kwargs])

    def __len__(self):
        return len(self.sets)

    def __call__(self, *args, ax=None, outfile=None, projection=None,
                 clear=True, **kwargs):
        if len(self.sets) < 1:
            raise IndexError('No datasets added')
        figsize = kwargs.pop('figsize', (7 * len(self.sets), 8))
        if ax is None:
            plt.figure(figsize=figsize)
        for i, set in enumerate(self.sets):
            if ax is not None:
                plot_axis = ax[i]
                plt.sca(ax[i])
            else:
                plt.subplot(1, len(self.sets), i + 1, projection=projection)
                plot_axis = plt.gca()
            kw = self.func(*set[0], **set[1])
            if i > 0:
                plt.ylabel('')
                try:
                    plot_axis.get_legend().remove()
                except AttributeError:
                    pass
        if outfile is not None:
            save_kwargs = kw if kw is not None else {}
            plt.savefig(outfile, **save_kwargs)
        if clear:
            self.sets = []


@_figure_wrapper
def plot_drawn_arms(drawn_arms, image_arr=None):
    if image_arr is not None:
        plt.imshow(image_arr, **_imshow_kwargs)
    for arm in drawn_arms:
        plt.plot(*arm.T)
    plt.axis('off')
    plt.gcf().subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.tight_layout()
    return dict(bbox_inches='tight', pad_inches=0)


@_figure_wrapper
def plot_clustered_points(drawn_arms, labels, image_arr=None):
    if image_arr is not None:
        plt.imshow(image_arr, **_imshow_kwargs)

    for l in np.unique(labels):
        for arm in drawn_arms[labels == l]:
            if l < 0:
                continue
            plt.plot(*arm.T, '.', c='C{}'.format(l % 10), markersize=2,
                     alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    return dict(bbox_inches='tight', pad_inches=0)


@FigureWrapper
def make_cleaning_plot(R, t, mask):
    plt.xlabel('Distance from center')
    plt.ylabel(r'$\theta$ (unwrapped)')
    plt.plot(R[~mask], t[~mask], 'r.', label='Points removed',
             markersize=4, alpha=0.5)
    plt.plot(R[mask], t[mask], '.', label='Points kept',
             markersize=4, alpha=0.5)
    plt.legend()


@FigureWrapper
def make_polar_cleaning_plot(R, t, mask):
    plt.plot(t[~mask], R[~mask], 'r.', label='Points removed',
             markersize=4, alpha=0.5)
    plt.plot(t[mask], R[mask], '.', label='Points kept',
             markersize=4, alpha=0.5)
    plt.legend()


@FigureWrapper
def make_point_weight_plot(R, weights):
    a = np.argsort(R.reshape(-1))
    plt.plot(R.reshape(-1)[a], weights[a])
    plt.xlabel('Distance from center')
    plt.ylabel('Point weight')


@FigureWrapper
def make_fit_plot(R, t, R_predict, fits):
    plt.plot(t, R, '.', markersize=3, alpha=0.4)
    for name, t_fit in fits:
        plt.plot(t_fit, R_predict, label=name)
    plt.legend()


@FigureWrapper
def make_fit_comparison_plot(data):
    d = list(data.items())
    scores = [i[1] for i in d]
    plt.boxplot(scores, showmeans=True, showfliers=False)
    r = ('poly_spiral_([0-9]+)', 'log_spiral')
    repl = (r'Polynomial (k=\1)', 'Log Spiral')
    labels = [re.sub(r[0], repl[0], i[0]) for i in d]
    labels = [re.sub(r[1], repl[1], s) for s in labels]
    plt.xticks(range(1, len(d) + 1), labels, rotation='vertical')
    plt.xlabel('Model')
    plt.ylabel(r'Adjusted-$\mathrm{R}^2$ score')
    plt.gcf().subplots_adjust(bottom=0.4)


def make_pipeline_plots(
    *args,
    image_arr=None, file_loc=None, deprojected_array=None, verbose=False,
    **kwargs
):
    gen = pipeline._pipeline_iterator(*args, **kwargs)
    v = verbose
    plot_drawn_arms(args[0], outfile='{}/drawn_arms'.format(file_loc),
                    image_arr=image_arr)
    _vprint(v, '1 - equalising arm length')
    drawn_arms = next(gen)
    _vprint(v, '2 - calculating distance matrix')
    distances = next(gen)
    _vprint(v, '3 - clustering arms')
    labels = next(gen)
    plot_clustered_points(drawn_arms, labels,
                          outfile='{}/arm_clusters'.format(file_loc),
                          image_arr=image_arr)
    out = []
    for i in range(np.max(labels) + 1):
        _vprint(v, 'Arm', i)
        _vprint(v, '\t1 - deprojecting arm')
        deprojected_coords = next(gen)
        _vprint(v, '\t2 - cleaning points')
        R_all, t_all_unwrapped, outlier_mask = next(gen)
        R, t = R_all[outlier_mask], t_all_unwrapped[outlier_mask]
        make_cleaning_plot.add(
            R_all, t_all_unwrapped, outlier_mask
        )
        make_polar_cleaning_plot.add(
            R_all, t_all_unwrapped, outlier_mask
        )
        _vprint(v, '\t3 - calculating weights')
        point_weights = next(gen)
        make_point_weight_plot.add(R, point_weights)
        _vprint(v, '\t4 - performing group cross-validation')
        out_ = next(gen)
        out.append(out_)

        # Calculate the models for plotting
        R_predict = np.linspace(min(R), max(R), 300)
        logsp_model = fitting.get_polynomial_pipeline(1)
        logsp_model.fit(np.log(R).reshape(-1, 1), t,
                        bayesianridge__sample_weight=point_weights)

        fits = [(
            'log_spiral',
            logsp_model.predict(np.log(R_predict.reshape(-1, 1)))
        )]

        for degree in range(1, 9):
            poly_model = fitting.get_polynomial_pipeline(degree)
            poly_model.fit(R.reshape(-1, 1), t,
                           bayesianridge__sample_weight=point_weights)
            fits += [(
                'Polynomial (degree {})'.format(degree),
                poly_model.predict(R_predict.reshape(-1, 1))
            )]

        make_fit_plot.add(R, t, R_predict, fits)
        out.append(out_)
        make_fit_comparison_plot.add(out_)

    # plot all the cached data
    try:
        make_cleaning_plot(outfile='{}/cleaning'.format(file_loc), clear=True)
        plt.close()
        make_polar_cleaning_plot(outfile='{}/cleaning_polar'.format(file_loc),
                                 projection='polar', clear=True)
        plt.close()
        make_point_weight_plot(outfile='{}/point_weights'.format(file_loc),
                               clear=True)
        plt.close()
        make_fit_plot(outfile='{}/fits'.format(file_loc), projection='polar',
                      clear=True)
        plt.close()
        make_fit_comparison_plot(outfile='{}/fit_comparison'.format(file_loc),
                                 clear=True)
        plt.close()

    except IndexError:
        print('No arm clusters found')

    return out

#
# def make_pipeline_plots(
#     drawn_arms, image_arr=None, image_size=512, phi=0, ba=1, distances=None,
#     clean_points=False, file_loc=None, deprojected_array=None, verbose=False
# ):
#     v = verbose
#     file_loc = file_loc if file_loc is not None else './'
#     if not os.path.isdir(file_loc):
#         os.mkdir(file_loc)
#     plot_drawn_arms(drawn_arms, outfile='{}/drawn_arms'.format(file_loc),
#                     image_arr=image_arr)
#     _vprint(v, '1 - equalising arm length')
#     drawn_arms = np.array(equalize_arm_length(np.array(drawn_arms)))
#     _vprint(v, '2 - calculating distance matrix')
#     if distances is None:
#         distances = metric.calculate_distance_matrix(drawn_arms)
#     _vprint(v, '3 - clustering arms')
#     db = clustering.cluster_arms(distances)
#     plot_clustered_points(drawn_arms, db.labels_,
#                           outfile='{}/arm_clusters'.format(file_loc),
#                           image_arr=image_arr)
#     out = []
#     for i in range(np.max(db.labels_) + 1):
#         _vprint(v, 'Arm', i)
#         arms = drawn_arms[db.labels_ == i]
#         coords, groups_all = cleaning.get_grouped_data(arms)
#         _vprint(v, '\t1 - deprojecting arm')
#         deprojected_coords = deprojecting.deproject_arm(
#             phi, ba,
#             coords / image_size - 0.5
#         )
#         R_all, t_all = r_theta_from_xy(*deprojected_coords.T)
#         t_all_unwrapped = fitting.unwrap(t_all, groups_all)
#         _vprint(v, '\t2 - cleaning points')
#         if clean_points:
#             outlier_mask = cleaning.clean_arms_polar(R_all, t_all_unwrapped,
#                                                      groups_all)
#         else:
#             outlier_mask = np.ones(R_all.shape[0], dtype=bool)
#         make_cleaning_plot.add(
#             R_all, t_all_unwrapped, outlier_mask
#         )
#         make_polar_cleaning_plot.add(
#             R_all, t_all_unwrapped, outlier_mask
#         )
#         groups = groups_all[outlier_mask]
#         R = R_all[outlier_mask].reshape(-1, 1)
#         t = t_all_unwrapped[outlier_mask]
#         _vprint(v, '\t3 - calculating weights')
#         point_weights = get_sample_weight(R.reshape(-1), groups)
#         make_point_weight_plot.add(R, point_weights)
#         _vprint(v, '\t4 - performing group cross-validation')
#         gkf = GroupKFold(n_splits=min(3, len(np.unique(groups))))
#         logsp_model = fitting.get_polynomial_pipeline(1)
#         out_ = {}
#         s = fitting.weighted_group_cross_val(
#             logsp_model,
#             np.log(R), t,
#             cv=gkf,
#             groups=groups,
#             weights=point_weights
#         )
#         out_['Log spiral'] = fitting.adjusted_r2(
#             s, t.shape[0], 2
#         )
#         for degree in range(1, 9):
#             poly_model = fitting.get_polynomial_pipeline(degree)
#             s = fitting.weighted_group_cross_val(
#                 poly_model,
#                 R, t,
#                 cv=gkf,
#                 groups=groups,
#                 weights=point_weights
#             )
#             out_['poly_spiral_{}'.format(degree)] = fitting.adjusted_r2(
#                 s, t.shape[0], degree
#             )
#
#         # calculate the models for plotting
#         R_predict = np.linspace(min(R), max(R), 300)
#
#         logsp_model.fit(np.log(R), t,
#                         bayesianridge__sample_weight=point_weights)
#         fits = [(
#             'log_spiral',
#             logsp_model.predict(np.log(R_predict.reshape(-1, 1)))
#         )]
#
#         for degree in range(1, 9):
#             poly_model = fitting.get_polynomial_pipeline(degree)
#             poly_model.fit(R, t, bayesianridge__sample_weight=point_weights)
#             fits += [(
#                 'Polynomial (degree {})'.format(degree),
#                 poly_model.predict(R_predict.reshape(-1, 1))
#             )]
#
#         make_fit_plot.add(R, t, R_predict, fits)
#         out.append(out_)
#         make_fit_comparison_plot.add(out_)
#     try:
#         make_cleaning_plot(outfile='{}/cleaning'.format(file_loc), clear=True)
#         plt.close()
#         make_polar_cleaning_plot(outfile='{}/cleaning_polar'.format(file_loc),
#                                  projection='polar', clear=True)
#         plt.close()
#         make_point_weight_plot(outfile='{}/point_weights'.format(file_loc),
#                                clear=True)
#         plt.close()
#         make_fit_plot(outfile='{}/fits'.format(file_loc), projection='polar',
#                       clear=True)
#         plt.close()
#         make_fit_comparison_plot(outfile='{}/fit_comparison'.format(file_loc),
#                                  clear=True)
#         plt.close()
#
#     except IndexError:
#         print('No arm clusters found')
#     return out
