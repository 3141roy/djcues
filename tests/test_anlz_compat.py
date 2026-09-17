"""rekordbox 7 PQT2 tags carry version 0x02000002; parsing must not fail."""

from construct import Int32ub

import djcues.db  # noqa: F401  # applies the PQT2 patch on import
from pyrekordbox.anlz import structs


def test_pqt2_accepts_rekordbox7_version():
    header = (
        b"\x00" * 4
        + Int32ub.build(0x02000002)
        + b"\x00" * 4
        + b"\x00" * 16  # 2 bpm ticks
        + Int32ub.build(0)  # entry_count
        + b"\x00" * 12
    )
    assert structs.PQT2.parse(header).u1 == 0x02000002
