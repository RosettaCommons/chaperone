"""Focused tests for the ESMFold2 prediction block."""

from blocks.networks.prediction.esmfold2 import ESMFold2Block


def test_compile_uses_stable_block_name_for_all_samples(tmp_path):
    outputdir = tmp_path / "outputs"
    outputdir.mkdir()
    first = outputdir / "toy_sample0.cif"
    second = outputdir / "toy_sample1.cif"
    first.write_text("data_first\n")
    second.write_text("data_second\n")
    (outputdir / "summary.csv").write_text(
        "name,length,mean_plddt,ptm,iptm\n"
        "toy_sample0,10,0.7,0.6,0.5\n"
        "toy_sample1,10,0.8,0.7,0.6\n"
    )

    result = ESMFold2Block(command_template=["true"]).compile(tmp_path).unpack()

    assert result["esmfold2.path"].tolist() == [str(first), str(second)]
    assert result["esmfold2.plddt"].tolist() == [0.7, 0.8]
