from pathlib import Path

import yaml

from distill_ood_detection.config import DatasetConfig, load_config
from distill_ood_detection.datasets.openood_cifar import (
    OPENOOD_CIFAR_ID_TEST_SIZE,
    build_openood_cifar_dataset_config,
    expected_openood_cifar_sizes,
)


def test_cifar10_protocol_uses_fixed_openood_manifests(tmp_path: Path) -> None:
    dataset = build_openood_cifar_dataset_config(
        DatasetConfig(name="cifar10", data_dir="unused"),
        tmp_path,
    )

    assert dataset.image_lists["test"].imglist_path == str(
        tmp_path / "data/benchmark_imglist/cifar10/test_cifar10.txt"
    )
    assert [(item.name, item.group) for item in dataset.ood_datasets] == [
        ("cifar100", "near"),
        ("tin", "near"),
        ("mnist", "far"),
        ("svhn", "far"),
        ("texture", "far"),
        ("places365", "far"),
    ]
    assert expected_openood_cifar_sizes("cifar10") == {
        "cifar10_test": OPENOOD_CIFAR_ID_TEST_SIZE,
        "cifar100_test": 9_000,
        "tin_test": 7_793,
        "mnist_test": 70_000,
        "svhn_test": 26_032,
        "texture_test": 5_640,
        "places365_test": 35_195,
    }


def test_selected_variants_preserve_requested_training_and_inference() -> None:
    selection_path = Path(
        "configs/evaluation/openood_cifar_v1_5/selected_variants.yaml"
    )
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))

    assert len(selection["variants"]) == 14
    for variant in selection["variants"]:
        config = load_config(Path(variant["config"]))
        assert config.dataset.name in {"cifar10", "cifar100"}
        if "aggressive_spatial" in variant["name"]:
            assert config.strategy.perturbation.teacher_target == "clean"
            assert config.strategy.perturbation.evaluation_draws == 50
            assert variant["apply_perturbation"] is True
        if "feature_denoising" in variant["name"]:
            assert config.strategy.feature_denoising.mask_probability == 0.2
            assert config.strategy.feature_denoising.evaluation_draws == 10
