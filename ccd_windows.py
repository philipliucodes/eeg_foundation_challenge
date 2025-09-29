from pathlib import Path
from typing import Tuple, Sequence, Optional

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)


def build_ccd_windows(cache_root: Path, mini: bool = False) -> BaseConcatDataset:
    releases = [f"R{i}" for i in range(1, 12)]
    datasets = []

    for r in releases:
        cache_r = Path(cache_root) / f"{r}_L100_bdf"
        assert (cache_r / "dataset_description.json").exists(), f"Missing {cache_r}"
        ds = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=r,
            cache_dir=cache_r,
            mini=mini,
        )
        datasets.append(ds)

    dataset_ccd = BaseConcatDataset(datasets)

    EPOCH_LEN_S = 2.0
    SFREQ = 100

    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset_ccd, transformation_offline, n_jobs=1)

    ANCHOR = "stimulus_anchor"
    SHIFT_AFTER_STIM = 0.5
    WINDOW_LEN_S = 2.0

    dataset = keep_only_recordings_with(ANCHOR, dataset_ccd)

    single_windows = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN_S) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
        verbose=False,
    )

    single_windows = add_extras_columns(
        single_windows,
        dataset,
        desc=ANCHOR,
        keys=(
            "target",
            "rt_from_stimulus",
            "rt_from_trialstart",
            "stimulus_onset",
            "response_onset",
            "correct",
            "response_type",
        ),
    )
    return single_windows


def split_ccd_by_subject(
    single_windows: BaseConcatDataset,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 2025,
    sub_rm: Optional[Sequence[str]] = [
        "NDARWV769JM7",
        "NDARME789TD2",
        "NDARUA442ZVF",
        "NDARJP304NK1",
        "NDARTY128YLU",
        "NDARDW550GU6",
        "NDARLD243KRE",
        "NDARUJ292JXV",
        "NDARBA381JGH",
    ],
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    meta_information = single_windows.get_metadata()

    subjects = meta_information["subject"].unique()
    subjects = [s for s in subjects if s not in sub_rm]

    train_subj, valid_test_subject = train_test_split(
        subjects,
        test_size=(valid_frac + test_frac),
        random_state=check_random_state(seed),
        shuffle=True,
    )

    valid_subj, test_subj = train_test_split(
        valid_test_subject,
        test_size=test_frac,
        random_state=check_random_state(seed + 1),
        shuffle=True,
    )

    assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)

    subject_split = single_windows.split("subject")
    train_set, valid_set, test_set = [], [], []

    for s in subject_split:
        if s in train_subj:
            train_set.append(subject_split[s])
        elif s in valid_subj:
            valid_set.append(subject_split[s])
        elif s in test_subj:
            test_set.append(subject_split[s])

    train_set = BaseConcatDataset(train_set)
    valid_set = BaseConcatDataset(valid_set)
    test_set = BaseConcatDataset(test_set)

    print("Number of examples in each split")
    print(f"Train:\t{len(train_set)}")
    print(f"Valid:\t{len(valid_set)}")
    print(f"Test:\t{len(test_set)}")

    return train_set, valid_set, test_set
