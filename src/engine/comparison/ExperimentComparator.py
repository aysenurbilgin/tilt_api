from __future__ import division

from bokeh.embed import components
from bokeh.charts import HeatMap
from bokeh.layouts import gridplot

from collections import OrderedDict

__author__ = 'abilgin'

class ExperimentComparator:

    def __init__(self, experiments):
        self.experiments = experiments

    def analogyEvalComparison(self):
        # construct data y axis experiment names, x axis years
        data = {}
        data['experiments'] = []
        data['timestamps'] = []
        data['accuracies'] = []
        max_intervals = []

        all_init_flag = True
        any_init_flag = False

        # explore the experiments wrt initialised to find the maximum number of bins
        for experiment in self.experiments:
            all_init_flag &= experiment.init_flag
            any_init_flag |= experiment.init_flag

            if not experiment.init_flag and not max_intervals:
                for timestamp in experiment.analogy_accuracies.keys():
                    max_intervals.append(timestamp)

        max_intervals.sort()

        for experiment in self.experiments:
            analogy_accuracies = OrderedDict(sorted(experiment.analogy_accuracies.items()))
            if not any_init_flag or all_init_flag:
                data['experiments'].extend([experiment.display_title]*len(analogy_accuracies.keys()))
                for timestamp in analogy_accuracies.keys():
                    total = analogy_accuracies[timestamp]["total"]
                    sum_corr = analogy_accuracies[timestamp]["correct"]
                    data['accuracies'].extend([(sum_corr / total) * 100])
                    data['timestamps'].extend([timestamp])
            else:
                if experiment.init_flag:
                    data['experiments'].extend([experiment.display_title]*len(max_intervals))
                    data['timestamps'].extend(max_intervals)
                    for timestamp in max_intervals:
                        if timestamp in analogy_accuracies.keys():
                            total = analogy_accuracies[timestamp]["total"]
                            sum_corr = analogy_accuracies[timestamp]["correct"]
                        else:
                            total = analogy_accuracies.values()[0]["total"]
                            sum_corr = analogy_accuracies.values()[0]["correct"]
                        data['accuracies'].extend([(sum_corr / total) * 100])
                else:
                    data['experiments'].extend([experiment.display_title]*len(analogy_accuracies.keys()))
                    for timestamp in analogy_accuracies.keys():
                        total = analogy_accuracies[timestamp]["total"]
                        sum_corr = analogy_accuracies[timestamp]["correct"]
                        data['accuracies'].extend([(sum_corr / total) * 100])
                        data['timestamps'].extend([timestamp])

        hm = HeatMap(data, x='timestamps', y='experiments', values='accuracies',
             title='Analogy Evaluation (Percentage)', stat=None)
        hm.legend.location = "bottom_right"
        hm.legend.background_fill_alpha = 0.7

        return [hm]

    def similarityEvalComparison(self):

        max_intervals = []
        methods_list = []
        hm_list = []

        all_init_flag = True
        any_init_flag = False

        # explore the experiments wrt initialised to find the maximum number of bins
        for experiment in self.experiments:
            all_init_flag &= experiment.init_flag
            any_init_flag |= experiment.init_flag

            if not experiment.init_flag and not max_intervals:
                for method in experiment.similarity_accuracies:
                    for timestamp in experiment.similarity_accuracies[method].keys():
                        max_intervals.append(timestamp)
                    break

            if not methods_list:
                for method in experiment.similarity_accuracies:
                    methods_list.append(method)

        max_intervals.sort()

        for method in methods_list:
            # construct data y axis experiment names, x axis years for each method
            data = {}
            data['experiments'] = []
            data['timestamps'] = []
            data['correlations'] = []

            for experiment in self.experiments:
                sim_accuracies = OrderedDict(sorted(experiment.similarity_accuracies[method].items()))

                if not any_init_flag or all_init_flag:
                    data['experiments'].extend([experiment.display_title] * len(sim_accuracies.keys()))
                    for timestamp in sim_accuracies.keys():
                        spearman_rho = sim_accuracies[timestamp]["spearman_rho"]
                        data['correlations'].extend([spearman_rho])
                        data['timestamps'].extend([timestamp])
                else:
                    if experiment.init_flag:
                        data['experiments'].extend([experiment.display_title] * len(max_intervals))
                        data['timestamps'].extend(max_intervals)
                        for timestamp in max_intervals:
                            if timestamp in sim_accuracies.keys():
                                spearman_rho = sim_accuracies[timestamp]["spearman_rho"]
                            else:
                                spearman_rho = sim_accuracies.values()[0]["spearman_rho"]
                            data['correlations'].extend([spearman_rho])
                    else:
                        data['experiments'].extend([experiment.display_title] * len(sim_accuracies.keys()))
                        for timestamp in sim_accuracies.keys():
                            spearman_rho = sim_accuracies[timestamp]["spearman_rho"]
                            data['correlations'].extend([spearman_rho])
                            data['timestamps'].extend([timestamp])

            hm = HeatMap(data, x='timestamps', y='experiments', values='correlations',
                         title="Similarity Evaluation using "+method+" (Spearman Rho)", stat=None)
            hm.legend.location = "bottom_right"
            hm.legend.background_fill_alpha = 0.7

            hm_list.append(hm)

        return hm_list

    def evalComparison(self):

        plots = []
        plots.extend(self.similarityEvalComparison())
        plots.extend(self.analogyEvalComparison())
        overview_layout = gridplot(plots, ncols=2, plot_height=80*len(self.experiments))
        script, div = components(overview_layout)
        return script, div



