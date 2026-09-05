from pathlib import Path

import pytest


@pytest.fixture
def input_files(tmp_path: Path) -> dict[str, Path]:
    expression = tmp_path / "expression.tsv"
    expression.write_text(
        "gene\tcell_1\tcell_2\tcell_3\tcell_4\n"
        "TF1\t1.0\t2.0\t4.0\t5.0\n"
        "TF2\t0.0\t1.0\t1.0\t3.0\n"
        "G3\t1.0\t3.0\t2.0\t6.0\n"
        "CONST\t2.0\t2.0\t2.0\t2.0\n",
        encoding="utf-8",
    )
    tf_list = tmp_path / "tf_list.txt"
    tf_list.write_text("TF1\nTF2\n", encoding="utf-8")
    groups = tmp_path / "groups.tsv"
    groups.write_text(
        "sample\tcluster\ncell_1\tA\ncell_2\tA\ncell_3\tB\ncell_4\tB\n",
        encoding="utf-8",
    )
    return {"expression": expression, "tf_list": tf_list, "groups": groups}
