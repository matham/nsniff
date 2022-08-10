import numpy as np
from math import ceil
from numpy import linalg as LA
from sklearn.decomposition import PCA
import nixio as nix
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from itertools import chain
from copy import deepcopy

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class SensorArray:

    name: str = ''

    times: np.ndarray = None

    data: np.ndarray = None

    def __init__(self, times, data, name):
        self.times = times
        self.data = data
        self.name = name

    def get_time_index(self, t):
        return np.sum(self.times < t)

    def get_interval(self, t_start, t_end) -> 'SensorArray':
        s = self.get_time_index(t_start)
        e = self.get_time_index(t_end)
        return SensorArray(self.times[s:e], self.data[s:e, :], self.name)

    def resample_to_samples(self, times):
        self.data = self.data[:len(times), :]
        self.times = self.times[:len(times)]

    def __deepcopy__(self, memodict={}):
        return SensorArray(self.times.copy(), self.data.copy(), self.name)


class Events:

    times: np.ndarray = None

    labels: List[str] = None

    key: str = ''

    def __init__(self, times, labels, key):
        self.times = times
        self.labels = labels
        self.key = key

    def parse_event_times(self, end_code) -> List[Tuple[str, float, float]]:
        t_events = self.times
        events = self.labels

        trials = []
        start = None
        for i, c in enumerate(events):
            if not c:
                continue

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
                trials.append((*start, t_events[i]))
                start = None

        if start is not None:
            raise ValueError('Last trial does not have end code')

        return trials


class Trial:

    trial_code: str = ''

    trial_key: str = ''

    experiment_data: List[SensorArray] = []

    trial_start: float = 0

    trial_end: float = 0

    pre: List[SensorArray] = []

    trial: List[SensorArray] = []

    post: List[SensorArray] = []

    def __init__(
            self, experiment_data, trial_code, trial_key, trial_start,
            trial_end):
        self.experiment_data = experiment_data
        self.trial_code = trial_code
        self.trial_key = trial_key
        self.trial_start = trial_start
        self.trial_end = trial_end

    def parse_trial(self, pre_trial_duration, post_trial_duration) -> None:
        self.pre = []
        self.post = []

        if pre_trial_duration:
            self.pre = [
                sensor.get_interval(
                    self.trial_start - pre_trial_duration, self.trial_start)
                for sensor in self.experiment_data
            ]
        self.trial = [
            sensor.get_interval(self.trial_start, self.trial_end)
            for sensor in self.experiment_data
        ]
        if post_trial_duration:
            self.post = [
                sensor.get_interval(
                    self.trial_end, self.trial_end + post_trial_duration)
                for sensor in self.experiment_data
            ]

        for sensors in (self.pre, self.trial, self.post):
            i, _ = sorted(
                enumerate((len(s.times) for s in sensors)),
                key=lambda x: x[1])[-1]
            times = sensors[i].times

            for k, sensor in enumerate(sensors):
                if k != i:
                    sensor.resample_to_samples(times)

    def __deepcopy__(self, memodict={}):
        # todo handle previous trial copy
        trial = Trial(
            self.experiment_data, self.trial_code, self.trial_key,
            self.trial_start, self.trial_end
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
            return [np.concatenate(item, axis=0) for item in data]
        else:
            raise ValueError(
                f'Unknown section {section}. Valid values are all, pre, '
                f'trial, and post')


class EventTrials:

    trial_code: str = ''

    trial_key: str = ''

    trials: List[Trial] = []

    def __init__(self, trial_code, trial_key):
        self.trials = []
        self.trial_code = trial_code
        self.trial_key = trial_key

    def __deepcopy__(self, memodict={}):
        trials = EventTrials(self.trial_code, self.trial_key)
        trials.trials = deepcopy(self.trials)
        return trials

    def collate_trial_data(self, section='all') -> List[List[np.ndarray]]:
        if not self.trials:
            raise ValueError('No trials')

        # get list for each sensor array
        sensors = [[] for _ in self.trials[0].trial]
        for trial in self.trials:
            # add all sensors to corresponding sensor for each trial
            for i, data in enumerate(trial.get_section(section)):
                sensors[i].append(data)
        return sensors

    @staticmethod
    def flatten_sensors_for_trial(data: List[np.ndarray]):
        mn = min((item.shape[0] for item in data))
        return np.concatenate(list(item[:mn, :] for item in data), axis=1)

    def flatten_collated_data(
            self, data: List[List[np.ndarray]]) -> List[np.ndarray]:
        # convert to (Time/Trial)xSensor, where S is sensors and time/trial is
        # stacked along columns
        return [np.concatenate(item, axis=0) for item in data]

    def stack_collated_data(
            self, data: List[List[np.ndarray]]) -> List[np.ndarray]:
        shapes = [[item.shape[0] for item in sensor] for sensor in data]
        min_shapes = min(min(items) for items in shapes)
        return [
            np.concatenate(
                [item[:min_shapes, :, np.newaxis] for item in sensor], axis=2
            )
            for sensor in data]

    def normalize_across_trials(
            self, baseline, sections: List[str]):
        collated = self.collate_trial_data(baseline)
        sensor_data = self.flatten_collated_data(collated)
        EventTrials.norm_trials_from_params(sensor_data, self.trials, sections)

    @staticmethod
    def normalize_across_trials_across_odors(
            trials: List[Trial], baseline, sections: List[str]):
        # get list for each sensor array
        collated = [[] for _ in trials[0].trial]
        for trial in trials:
            # add all sensors to corresponding sensor for each trial
            vals = trial.get_section(baseline)
            mn = min((item.shape[0] for item in vals))
            for i, data in enumerate(vals):
                collated[i].append(data[:mn, :])

        sensor_data = [np.concatenate(items, axis=0) for items in collated]
        EventTrials.norm_trials_from_params(sensor_data, trials, sections)

    @staticmethod
    def norm_trials_from_params(
            sensor_data: List[np.ndarray], trials: List[Trial],
            sections: List[str]):
        means = [np.mean(data, axis=0, keepdims=True) for data in sensor_data]
        norms = [LA.norm(data - mean, axis=0, keepdims=True)
                 for mean, data in zip(means, sensor_data)]

        sensor: SensorArray
        for trial in trials:
            for section in sections:
                for mean, norm, sensor in zip(
                        means, norms, getattr(trial, section)):
                    data = sensor.data - mean
                    mask = norm != 0
                    data[:, np.squeeze(mask)] /= norm[mask]
                    sensor.data = data

    def normalize_each_trial(self, baseline, sections: List[str]):
        for trial in self.trials:
            sensor_data = trial.get_section(baseline)
            norms = [LA.norm(data, axis=0) for data in sensor_data]

            sensor: SensorArray
            for section in sections:
                for norm, sensor in zip(norms, getattr(trial, section)):
                    sensor.data = sensor.data / norm[np.newaxis, :]


class Experiment:

    trials: List[Trial] = []

    runs: List[Tuple[List[SensorArray], List[Events]]] = []

    grouped_trials: List[EventTrials] = []

    end_code: str = ''

    def __init__(self, end_code=''):
        self.runs = []
        self.trials = []
        self.grouped_trials = []
        self.end_code = end_code

    def read_data(self, filename) -> None:
        f = nix.File.open(filename, mode=nix.FileMode.ReadOnly)
        datas: List[nix.DataArray] = f.blocks['experiment'].data_arrays

        experiments = []
        for item in datas:
            if item.name.startswith('experiment_names_'):
                exp_name = item.name.replace('_names', '')
                names = [s.decode() for s in np.asarray(item).tolist()]
                times = np.asarray(datas[exp_name])
                key = datas[exp_name].metadata['protocol_key']
                experiments.append(Events(times, names, key))

        sensors = []
        data: nix.DataArray
        for block in f.blocks:
            if block.name.startswith('stratuscent_dev_'):
                num = block.name[len('stratuscent_dev_'):]

                arr = np.asarray(block.data_arrays['sensor_0'])
                data = arr['data']

                time_conv = np.asarray(block.data_arrays['sensor_times_0'])
                local = (time_conv[:, 0] + time_conv[:, 2]) / 2
                remote = time_conv[:, 1]
                A = np.vstack([remote, np.ones(len(remote))]).T
                m, c = LA.lstsq(A, local, rcond=None)[0]
                times = m * arr['timestamp'] + c

                sensors.append(SensorArray(times, data, num))

        # sensors = sorted(sensors, key=lambda sensor: sensor.name)
        self.runs.append((sensors, experiments))

    def parse_trials(self, pre_trial_duration, post_trial_duration) -> None:
        trials = self.trials = []
        groups = {}
        for sensors, events in self.runs:
            for event in events:
                for label, ts, te in event.parse_event_times(self.end_code):
                    trial = Trial(sensors, label, event.key, ts, te)
                    trial.parse_trial(pre_trial_duration, post_trial_duration)
                    trials.append(trial)

                    if label not in groups:
                        groups[label] = EventTrials(label, event.key)
                    groups[label].trials.append(trial)

        self.grouped_trials = list(groups.values())

    @staticmethod
    def normalize_trials(
            grouped_trials: List[EventTrials], per_trial=True, baseline='all',
            sections_to_norm=('pre', 'trial', 'post'), per_odor=False
    ) -> List[EventTrials]:
        grouped_trials = deepcopy(grouped_trials)
        if per_odor:
            for trials in grouped_trials:
                if per_trial:
                    trials.normalize_each_trial(baseline, sections_to_norm)
                else:
                    trials.normalize_across_trials(baseline, sections_to_norm)
        else:
            trials = [
                trial for items in grouped_trials for trial in items.trials]
            EventTrials.normalize_across_trials_across_odors(
                trials, baseline, sections_to_norm)
        return grouped_trials

    def plot_groups_2d_grid(
            self, grouped_trials: List[EventTrials], section, n_cols,
            individual_trials=False, summary='mean'):
        fig, axs = plt.subplots(
            ceil(len(grouped_trials) / n_cols), n_cols, squeeze=False,
            sharey=True, sharex=True)
        if individual_trials:
            fig.suptitle('Individual trials')
        else:
            fig.suptitle(f'Trial {summary}')

        trials: EventTrials
        for ax, trials in zip(chain(*axs), grouped_trials):
            ax.set_title(f'{trials.trial_code} ({trials.trial_key})')
            if individual_trials:
                for i, trial in enumerate(trials.trials):
                    sensor_data = trial.get_section(section)
                    flat_data = EventTrials.flatten_sensors_for_trial(
                        sensor_data)
                    summarized = getattr(np, summary)(flat_data, axis=0)
                    ax.plot(summarized, '*', color=colors[i])
                    ax.set_xlabel('Sensor #')
            else:
                sensor_data = trials.flatten_collated_data(
                    trials.collate_trial_data(section))
                summarized = [
                    getattr(np, summary)(d, axis=0) for d in sensor_data]
                summarized = np.concatenate(summarized)
                ax.plot(summarized, '*')
                ax.set_xlabel('Sensor #')

    def plot_groups_2d(
            self, grouped_trials: List[EventTrials], section, summary='mean'):
        trials: EventTrials
        plt.figure()
        for trials in grouped_trials:
            sensor_data = trials.flatten_collated_data(
                trials.collate_trial_data(section))
            summarized = [
                getattr(np, summary)(d, axis=0) for d in sensor_data]
            summarized = np.concatenate(summarized)
            plt.plot(summarized, '*', label=trials.trial_code)
            plt.xlabel('Sensor #')
            plt.title(f'Trial {summary}')
            plt.legend()

    def plot_groups_3d(
            self, grouped_trials: List[EventTrials], section, n_cols,
            summary='mean', norm_each_sensor=True):
        fig, axs = plt.subplots(
            ceil(len(grouped_trials) / n_cols), n_cols, squeeze=False,
            sharey=True)
        fig.suptitle(f'Trial {summary}')

        trials: EventTrials
        for ax, trials in zip(chain(*axs), grouped_trials):
            ax.set_title(f'{trials.trial_code} ({trials.trial_key})')
            sensor_data = trials.stack_collated_data(
                trials.collate_trial_data(section))
            sensor_data = np.concatenate(sensor_data, axis=1)
            summarized = getattr(np, summary)(sensor_data, axis=2)

            if norm_each_sensor:
                summarized -= np.min(summarized, axis=0, keepdims=True)
                mx = np.max(summarized, axis=0, keepdims=True)
                mask = mx[0, :] != 0
                summarized[:, mask] /= mx[:, mask]

            ax.pcolormesh(summarized.T)
            ax.set_xlabel('Time')
            ax.set_ylabel('Sensor #')

    def remove_sensors(self, keep):
        runs = []
        for sensors, events in self.runs:
            runs.append(([sensors[keep]], events))
        self.runs = runs

    def compute_pca(self, n_components):
        items = []
        for sensors, _ in self.runs:
            mn = min((sensor.data.shape[0] for sensor in sensors))
            data = np.concatenate(
                [sensor.data[:mn, :] for sensor in sensors], axis=1)
            items.append(data)

        data = np.concatenate(items, axis=0)

        pca = PCA(n_components=n_components)
        pca.fit(data)
        print((pca.explained_variance_ratio_ * 100).tolist())

        runs = []
        for item, (sensors, events) in zip(items, self.runs):
            sensor = sensors[0]
            sensor.data = pca.transform(item)
            sensor.times = sensor.times[:item.shape[0]]
            runs.append(([sensor], events))
        self.runs = runs


if __name__ == '__main__':
    files = r'G:\Python\peanut2022_01.h5', r'G:\Python\peanut2022_02.h5'

    experiment = Experiment(end_code='end')
    for filename in files:
        experiment.read_data(filename)
    # experiment.compute_pca(10)
    # experiment.remove_sensors(5)
    experiment.parse_trials(pre_trial_duration=50, post_trial_duration=50)

    normalized_trials = experiment.normalize_trials(
        experiment.grouped_trials, per_trial=False, baseline='pre',
        per_odor=False)
    # normalized_trials = experiment.grouped_trials

    # experiment.plot_groups_2d_grid(
    #     normalized_trials, section='trial', n_cols=4, individual_trials=False)
    # experiment.plot_groups_2d_grid(
    #     normalized_trials, section='trial', n_cols=4, individual_trials=True)
    # experiment.plot_groups_2d(normalized_trials, section='trial')
    experiment.plot_groups_3d(normalized_trials, section='all', n_cols=4)

    plt.show()
