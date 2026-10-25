from lsgames.cli import main


def test_cli_reports_values(capsys):
    assert main(["magic-square", "--shots", "50", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "Classical value: 17/18" in out
    assert "Quantum win probability (exact): 1.000000" in out
