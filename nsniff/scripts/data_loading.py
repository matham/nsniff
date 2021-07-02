import numpy as np
from math import ceil
from numpy import linalg as LA
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from itertools import chain
from copy import deepcopy

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class SensorArray:

    times: np.ndarray = None

    data: np.ndarray = None

    def __init__(self, times, data):
        self.times = times
        self.data = data

    def get_time_index(self, t):
        return np.sum(self.times < t)

    def get_interval(self, t_start, t_end) -> 'SensorArray':
        s = self.get_time_index(t_start)
        e = self.get_time_index(t_end)
        return SensorArray(self.times[s:e], self.data[:, s:e])

    def __deepcopy__(self, memodict={}):
        return SensorArray(self.times.copy(), self.data.copy())


class Events:

    times: np.ndarray = None

    events: List[str] = None

    def __init__(self, times, events):
        self.times = times
        self.events = events

    def parse_event_times(self, end_code) -> List[Tuple[str, float, float]]:
        t_events = self.times
        events = self.events
        trials = []
        start = None
        for i, c in enumerate(events):
            if start is None:
                if c == end_code:
                    raise ValueError(
                        f'Got end code before start code at index {i}, '
                        f'time {t_events[i]}')
                start = events[i], t_events[i]
            else:
                if c != end_code:
                    raise ValueError(
                        f'Expected end code at index {i}, time {t_events[i]}')
                trials.append((start[0], start[1], t_events[i]))
                start = None

        if start is not None:
            raise ValueError('Last trial does not have end code')

        return trials


class Trial:

    previous_trial: 'Trial' = None

    trial_code: str = ''

    experiment_data: List[SensorArray] = []

    trial_start: float = 0

    trial_end: float = 0

    pre: List[SensorArray] = []

    trial: List[SensorArray] = []

    post: List[SensorArray] = []

    def __init__(
            self, experiment_data, trial_code, trial_start, trial_end,
            previous_trial=None):
        self.experiment_data = experiment_data
        self.trial_code = trial_code
        self.trial_start = trial_start
        self.trial_end = trial_end
        self.previous_trial = previous_trial

    def parse_trial(self, pre_trial_duration, post_trial_duration) -> None:
        self.pre = [
            sensor.get_interval(
                self.trial_start - pre_trial_duration, self.trial_start)
            for sensor in self.experiment_data
        ]
        self.trial = [
            sensor.get_interval(self.trial_start, self.trial_end)
            for sensor in self.experiment_data
        ]
        self.post = [
            sensor.get_interval(
                self.trial_end, self.trial_end + post_trial_duration)
            for sensor in self.experiment_data
        ]

    def __deepcopy__(self, memodict={}):
        # todo handle previous trial copy
        trial = Trial(
            self.experiment_data, self.trial_code, self.trial_start,
            self.trial_end
        )
        trial.pre = deepcopy(self.pre)
        trial.trial = deepcopy(self.trial)
        trial.post = deepcopy(self.post)
        return trial

    def get_section(self, section) -> List[np.ndarray]:
        if section in ('pre', 'trial', 'post'):
            return [s.data for s in getattr(self, section)]
        elif section == 'all':
            data = [[] for _ in self.trial]
            for i, sensor in enumerate(self.pre):
                data[i].append(sensor.data)
            for i, sensor in enumerate(self.trial):
                data[i].append(sensor.data)
            for i, sensor in enumerate(self.post):
                data[i].append(sensor.data)
            return [np.concatenate(item, axis=1) for item in data]
        else:
            raise ValueError(
                f'Unknown section {section}. Valid values are all, pre, '
                f'trial, and post')


class EventTrials:

    trial_code: str = ''

    trials: List[Trial] = []

    def __init__(self, trial_code):
        self.trials = []
        self.trial_code = trial_code

    def __deepcopy__(self, memodict={}):
        trials = EventTrials(self.trial_code)
        trials.trials = deepcopy(self.trials)
        return trials

    def collate_trial_data(self, section='all') -> List[List[np.ndarray]]:
        if not self.trials:
            raise ValueError('No trials')

        sensors = [[] for _ in self.trials[0].trial]
        for trial in self.trials:
            for i, data in enumerate(trial.get_section(section)):
                sensors[i].append(data)
        return sensors

    def flatten_collated_data(
            self, data: List[List[np.ndarray]]) -> List[np.ndarray]:
        return [np.concatenate(item, axis=1) for item in data]

    def stack_collated_data(
            self, data: List[List[np.ndarray]]) -> List[np.ndarray]:
        shapes = [[item.shape[1] for item in sensor] for sensor in data]
        min_shapes = [min(items) for items in shapes]
        return [
            np.concatenate(
                [item[:, :shape, np.newaxis] for item in sensor], axis=2
            )
            for shape, sensor in zip(min_shapes, data)]

    def normalize_across_trials(
            self, baseline, sections: List[str]):
        collated = self.collate_trial_data(baseline)
        sensor_data = self.flatten_collated_data(collated)
        norms = [LA.norm(data, axis=1) for data in sensor_data]

        sensor: SensorArray
        for trial in self.trials:
            for section in sections:
                for norm, sensor in zip(norms, getattr(trial, section)):
                    sensor.data = sensor.data / norm[:, np.newaxis]

    def normalize_each_trial(self, baseline, sections: List[str]):
        for trial in self.trials:
            sensor_data = trial.get_section(baseline)
            norms = [LA.norm(data, axis=1) for data in sensor_data]

            sensor: SensorArray
            for section in sections:
                for norm, sensor in zip(norms, getattr(trial, section)):
                    sensor.data = sensor.data / norm[:, np.newaxis]


class Experiment:

    trials: List[Trial] = []

    runs: List[Tuple[List[SensorArray], Events]] = [([], None)]

    end_code: str = ''

    grouped_trials: List[EventTrials] = []

    def __init__(self, end_code: str):
        self.runs = []
        self.trials = []
        self.end_code = end_code
        self.grouped_trials = []

    def read_data(self, filename) -> None:
        data = pd.read_csv(filename)

        t_1 = data.iloc[:, 0].to_numpy()
        array_1 = data.iloc[:, 1:33].to_numpy().T
        not_nan = np.logical_not(np.isnan(t_1))
        t_1 = t_1[not_nan]
        array_1 = array_1[:, not_nan]

        t_2 = data.iloc[:, 37].to_numpy()
        array_2 = data.iloc[:, 38:70].to_numpy().T
        not_nan = np.logical_not(np.isnan(t_2))
        t_2 = t_2[not_nan]
        array_2 = array_2[:, not_nan]

        t_event = data.iloc[:, 74].to_numpy()
        events = data.iloc[:, 75].to_numpy()
        not_nan = np.logical_not(np.isnan(t_event))
        t_event = t_event[not_nan]
        events = events[not_nan].tolist()

        self.runs.append(
            ([SensorArray(t_1, array_1), SensorArray(t_2, array_2)],
             Events(t_event, events)))

    def parse_data(self, pre_trial_duration, post_trial_duration) -> None:
        trials = self.trials = []
        groups = {}
        for sensors, events in self.runs:
            trial = None
            trial_times = events.parse_event_times(self.end_code)
            for trial_code, ts, te in trial_times:
                trial = Trial(sensors, trial_code, ts, te, trial)
                trial.parse_trial(pre_trial_duration, post_trial_duration)
                trials.append(trial)

                if trial_code not in groups:
                    groups[trial_code] = EventTrials(trial_code)
                groups[trial_code].trials.append(trial)

        self.grouped_trials = list(groups.values())

    def get_num_sensors(self):
        if not self.trials:
            raise ValueError('No trials')
        trial = self.trials[0]
        if not trial.trial:
            raise ValueError('No trial data')
        return len(trial.trial)

    @staticmethod
    def normalize_trials(
            grouped_trials: List[EventTrials], per_trial=True, baseline='all',
            sections_to_norm=('pre', 'trial', 'post')
    ) -> List[EventTrials]:
        grouped_trials = deepcopy(grouped_trials)
        for trials in grouped_trials:
            if per_trial:
                trials.normalize_across_trials(baseline, sections_to_norm)
            else:
                trials.normalize_each_trial(baseline, sections_to_norm)
        return grouped_trials

    def plot_groups_2d(
            self, grouped_trials: List[EventTrials], section, n_cols,
            individual_trials=False, summary='mean'):
        fig, axs = plt.subplots(ceil(len(grouped_trials) / n_cols), n_cols)
        trials: EventTrials
        for ax, trials in zip(chain(*axs), grouped_trials):
            ax.set_title(trials.trial_code)
            if individual_trials:
                for trial in trials.trials:
                    sensor_data = trial.get_section(section)
                    summarized = [
                        getattr(np, summary)(d, axis=1) for d in sensor_data]
                    for i, sensor in enumerate(summarized):
                        ax.plot(sensor, '*', color=colors[i])
            else:
                sensor_data = trials.flatten_collated_data(
                    trials.collate_trial_data(section))
                summarized = [
                    getattr(np, summary)(d, axis=1) for d in sensor_data]
                for i, sensor in enumerate(summarized):
                    ax.plot(sensor, '*', color=colors[i])

    def plot_groups_3d(
            self, grouped_trials: List[EventTrials], sensor_num, section,
            n_cols, summary='mean'):
        fig, axs = plt.subplots(ceil(len(grouped_trials) / n_cols), n_cols)
        trials: EventTrials
        for ax, trials in zip(chain(*axs), grouped_trials):
            ax.set_title(trials.trial_code)
            sensor_data = trials.stack_collated_data(
                trials.collate_trial_data(section))
            summarized = getattr(np, summary)(sensor_data[sensor_num], axis=2)
            ax.pcolormesh(summarized)


if __name__ == '__main__':
    files = r'G:\peanut_1.csv', r'G:\peanut_2.csv', r'G:\peanut_3.csv'

    experiment = Experiment(end_code='x')
    for filename in files:
        experiment.read_data(filename)
    experiment.parse_data(pre_trial_duration=60, post_trial_duration=60)

    normalized_trials = experiment.normalize_trials(
        experiment.grouped_trials, per_trial=True, baseline='pre')
    # experiment.plot_groups_2d(
    #     normalized_trials, section='trial', n_cols=4, individual_trials=False)
    experiment.plot_groups_3d(
        normalized_trials, sensor_num=0, section='all', n_cols=4)

    plt.show()
