import numpy as np

import config
from level_data import load_game_assets, sanitize_levels_xml
from window_capture import is_zuma_process


def test_is_zuma_process_rejects_ide_and_terminals():
    assert is_zuma_process("Zuma.exe") is True
    assert is_zuma_process("popcapgame1.exe") is True
    assert is_zuma_process("zumadeluxe.exe") is True
    for non_game in ("Code.exe", "Antigravity.exe", "WindowsTerminal.exe", "explorer.exe", "chrome.exe", "python.exe", ""):
        assert is_zuma_process(non_game) is False


def test_sanitize_levels_xml_handles_unquoted_and_duplicates():
    raw = '<!-- comment -->\n<Settings id="level1" repeat="40" repeat="50" vx=-1 powerfreq=2500 />'
    cleaned = sanitize_levels_xml(raw)
    assert "comment" not in cleaned
    assert 'vx="-1"' in cleaned
    assert 'powerfreq="2500"' in cleaned
    assert cleaned.count("repeat=") == 1
    assert 'repeat="50"' in cleaned


def test_load_real_zuma_assets():
    assets = load_game_assets(config.DEFAULT_ZUMA_DIR)
    assert len(assets.levels) == 22

    total_curves = sum(len(lvl.curves) for lvl in assets.levels.values())
    assert total_curves == 25

    for dual_id in ("blackswirley", "serpents", "snakepit"):
        assert len(assets.levels[dual_id].curves) == 2

    assert assets.levels["space"].is_space is True
    assert assets.levels["space"].bg_bgr is None

    assert assets.levels["riverbed"].has_animated_bg is True
    assert assets.levels["serpents"].has_animated_bg is True

    # Verify tunnel flags and partial bridge visibility on underover and tunnellevel
    for tunnel_map in ("underover", "tunnellevel"):
        c = assets.levels[tunnel_map].curves[0]
        assert bool(np.any(c.in_tunnel))
        partial = (c.visibility > 0.05) & (c.visibility < 0.95)
        assert bool(np.any(partial))

    # Verify 24 powerup template sets (6 colors x 4 types x 24 rotated angles)
    assert len(assets.powerup_templates) == 6
    for col in config.COLORS:
        assert len(assets.powerup_templates[col]) == 4
        for ptype in config.POWERUPS:
            assert len(assets.powerup_templates[col][ptype]) == 24
            assert assets.powerup_templates[col][ptype][0].shape == (18, 18, 3)
