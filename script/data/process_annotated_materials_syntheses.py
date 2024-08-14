"""
Linearize materials science procedures from BRAT annotations.
"""

from sciriff.lib import paths, util
from sciriff.lib.bigbiohub import parse_brat_file, brat_parse_to_bigbio_kb


in_dir = paths.project_root / "data/preprocessing/downloads/annotated-materials-syntheses-master"
brat_dir = in_dir / "data"
out_dir = paths.project_root / "data/preprocessing/processed/annotated_materials_syntheses"
out_dir.mkdir(exist_ok=True, parents=True)


fold_map = {"train": "train", "dev": "validation", "test": "test"}

for fold in fold_map:
    out_fold_name = fold_map[fold]
    brat_names = [
        line.strip() for line in open(in_dir / f"sfex-{fold}-fnames.txt").readlines()
    ]

    examples = []

    for brat_name in brat_names:
        example = brat_parse_to_bigbio_kb(
            parse_brat_file(
                txt_file=brat_dir / f"{brat_name}.txt",
                annotation_file_suffixes=[".ann"],
            )
        )
        examples.append(example)

    util.write_jsonl(examples, out_dir / f"{out_fold_name}.jsonl")
