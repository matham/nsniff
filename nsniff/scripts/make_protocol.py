from typing import List, Optional
import csv
from random import shuffle


def get_index(
        i: Optional[int], name: Optional[str], names: List[str], value: float
) -> Optional[int]:
    if i is not None:
        return i
    if name is not None:
        return names.index(name)

    if value:
        raise ValueError(f'Value "{value}" given, but no valve/mfc')
    return None


def get_indicis(i: List[int], name: List[str], names: List[str]) -> List[int]:
    if i:
        return i
    if name:
        return [names.index(n) for n in name]
    return []


def get_indicis_value(
        i: List[int], name: List[str], names: List[str], values: List[float]
) -> List[int]:
    if i:
        if not values or len(i) != len(values):
            raise ValueError(f'Provided indices {i} but not enough values')
        return i
    if name:
        if not values or len(name) != len(values):
            raise ValueError(f'Provided names {i} but not enough values')
        return [names.index(n) for n in name]

    if values:
        raise ValueError(f'Provided values {i} but not enough indices/names')
    return []


class Trial:

    odor_names: List[str] = []

    odor_indicis: List[int] = []

    mfc_names: List[str] = []

    mfc_indicis: List[int] = []

    mfc_rate: List[float] = []

    key: str = ''

    label: str = ''

    duration: float = 0

    post_iti: float = 0

    pre_iti: float = 0

    mix_name: Optional[str] = None

    mix_index: Optional[int] = None

    mix_duration: float = 0

    def __init__(
            self, odor_names: List[str] = None,
            odor_indicis: List[int] = None, mfc_names: List[str] = None,
            mfc_indicis: List[int] = None, mfc_rate: List[float] = None,
            key: str = '', label: str = '',
            duration: float = 0, post_iti: float = 0, pre_iti: float = 0,
            mix_name: Optional[str] = None, mix_index: Optional[int] = None,
            mix_duration: float = 0, **kwargs
    ):
        super().__init__(**kwargs)
        self.odor_names = odor_names or []
        self.odor_indicis = odor_indicis or []
        self.mfc_names = mfc_names or []
        self.mfc_indicis = mfc_indicis or []
        self.mfc_rate = mfc_rate or []
        self.key = key
        self.label = label
        self.duration = duration
        self.post_iti = post_iti
        self.pre_iti = pre_iti
        self.mix_name = mix_name
        self.mix_index = mix_index
        self.mix_duration = mix_duration

    def get_trial_rows(
            self, valve_names: List[str], mfc_names: List[str]
    ) -> List[List[str]]:
        lines = []
        n_valves = len(valve_names)
        n_mfc = len(mfc_names)
        valves_zero = ['0', ] * n_valves
        mfcs_zero = ['0', ] * n_mfc
        key = self.key

        odors = get_indicis(self.odor_indicis, self.odor_names, valve_names)
        mfc_rate = self.mfc_rate
        used_mfc = get_indicis_value(
            self.mfc_indicis, self.mfc_names, mfc_names, mfc_rate)
        mix_dur = self.mix_duration
        mix_i = get_index(self.mix_index, self.mix_name, valve_names, mix_dur)

        valves_on = list(valves_zero)
        for i in odors:
            valves_on[i] = '1'
        valves_with_mix = list(valves_on)
        if mix_i is not None:
            valves_with_mix[mix_i] = '1'

        mfc_on = list(mfcs_zero)
        for i, value in zip(used_mfc, mfc_rate):
            mfc_on[i] = str(value)

        if self.pre_iti:
            lines.append(
                [str(self.pre_iti)] + valves_zero + mfcs_zero + [key, ''])

        if used_mfc:
            lines.append(['0'] + valves_on + mfcs_zero + [key, ''])
        if mix_i is not None:
            lines.append(
                [str(mix_dur)] + valves_on + mfc_on + [key, ''])
        lines.append(
            [str(self.duration)] +
            valves_with_mix + mfc_on + [key, self.label])

        label = 'end' if self.label else ''
        if used_mfc:
            lines.append(['0'] + valves_on + mfcs_zero + [key, label])
            label = ''

        if self.post_iti:
            lines.append(
                [str(self.post_iti)] + valves_zero + mfcs_zero + [key, label])
        elif label or odors:
            lines.append(['0'] + valves_zero + mfcs_zero + [key, label])

        return lines


class Protocol:

    valve_names: List[str] = []

    mfc_names: List[str] = []

    trials: List[Trial] = []

    def __init__(self, valve_names: List[str], mfc_names: List[str]):
        super().__init__()
        self.valve_names = valve_names
        self.mfc_names = mfc_names
        self.trials = []

    def add_trial(self, trial: Trial, repeat: int = 1):
        for _ in range(repeat):
            self.trials.append(trial)

    def add_iti_between_trials(self, duration: float, ending_iti: bool = True):
        trials = []
        for trial in self.trials:
            trials.append(trial)
            trials.append(Trial(duration=duration, key=trial.key))

        if not ending_iti and trials:
            del trials[-1]
        self.trials = trials

    def randomize_trial_order(self):
        shuffle(self.trials)

    def export_to_csv(self, filename):
        lines = []
        valve_names = self.valve_names
        mfc_names = self.mfc_names
        trials = self.trials

        header = ['duration']
        header.extend([f'valve_{s}' for s in valve_names])
        header.extend([f'mfc_{s}' for s in mfc_names])
        header.extend(['key', 'label'])

        trial_keys = sorted(set(t.key for t in trials))
        for key in trial_keys:
            for trial in trials:
                if trial.key == key:
                    lines.extend(trial.get_trial_rows(valve_names, mfc_names))

        with open(filename, 'w') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow(header)
            writer.writerows(lines)


if __name__ == '__main__':
    protocol = Protocol(
        valve_names=['mix', 'a', 'b', 'c', 'd', 'e', 'f', 'g'],
        mfc_names=['a;b;c', 'd;e;f;g']
    )

    protocol.add_trial(
        Trial(
            odor_names=['a'], mfc_names=['a;b;c'], mfc_rate=[.2], key='d',
            label='banana', duration=25, mix_name='mix', mix_duration=10),
        repeat=2
    )
    protocol.add_trial(
        Trial(
            odor_names=['d'], mfc_names=['d;e;f;g'], mfc_rate=[.2], key='d',
            label='apple', duration=25, mix_name='mix', mix_duration=10),
        repeat=2
    )
    protocol.add_trial(
        Trial(
            odor_names=['a', 'd'], mfc_names=['a;b;c', 'd;e;f;g'],
            mfc_rate=[.1, .2], key='d',
            label='banana:apple:33:66', duration=25, mix_name='mix',
            mix_duration=10),
        repeat=2
    )
    protocol.add_trial(
        Trial(
            odor_names=['a'], mfc_names=['a;b;c'], mfc_rate=[.2], key='a',
            label='banana', duration=25, mix_name='mix', mix_duration=10),
        repeat=3
    )

    protocol.randomize_trial_order()
    protocol.add_iti_between_trials(30)
    protocol.export_to_csv(r'G:\Python\experiment.csv')
