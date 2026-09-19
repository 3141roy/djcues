"""Cue placement strategy engine — maps PSSI phrases to cue points."""

from __future__ import annotations

from djcues.constants import CUE_SYSTEM, CUE_SYSTEM_BY_PAD
from djcues.models import BeatGrid, CuePoint, CueProposal, Phrase, Track


def _spectral_similarity(wf_points: list, i0: int, i_mid: int, i1: int) -> float:
    """Compare the RGB (bass/mid/treble) profile of two halves of a waveform section.

    Returns a similarity score from 0.0 (completely different) to 1.0 (identical).
    Uses mean-squared-error of the per-point RGB values, normalized.
    """
    half_len = min(i_mid - i0, i1 - i_mid)
    if half_len < 2:
        return 0.0

    first_half = wf_points[i0 : i0 + half_len]
    second_half = wf_points[i_mid : i_mid + half_len]

    total_mse = 0.0
    for a, b in zip(first_half, second_half):
        # RGB values are 0-7, normalize to 0-1
        dr = (a.red - b.red) / 7
        dg = (a.green - b.green) / 7
        db = (a.blue - b.blue) / 7
        dh = a.height - b.height  # already 0-1
        total_mse += dr * dr + dg * dg + db * db + dh * dh

    # Max possible MSE per point = 4 (all channels off by 1.0)
    mse = total_mse / half_len / 4
    return max(0.0, 1.0 - mse)


def _find_stable_loop(
    track: Track,
    search_start_ms: float,
    search_end_ms: float,
    bar_sizes: tuple[int, ...] = (8, 4, 2, 1),
    min_energy: float = 0.05,
    min_similarity: float = 0.7,
) -> tuple[float, int, float] | None:
    """Find a stable, loopable region using waveform energy and spectral similarity.

    Scans bar-aligned windows from search_start_ms forward. Tries 8 bars first,
    then falls back to 4, 2, 1. A good loop has non-trivial energy AND the
    second half spectrally matches the first half (so the loop repeats cleanly).

    Returns (position_ms, loop_bars, similarity_score) or None if nothing found.
    """
    if not track.waveform:
        return None

    bg = track.beat_grid
    n = len(track.waveform)
    total_ms = track.duration_ms
    if total_ms <= 0 or n == 0:
        return None

    for loop_bars in bar_sizes:
        loop_ms = bg.bars_to_ms(loop_bars)
        best_pos = None
        best_score = 0.0

        # Snap search start to bar boundary
        start_beat = bg.ms_to_beat(search_start_ms)
        bar_start = ((start_beat - 1) // 4) * 4 + 1
        pos_ms = bg.beat_to_ms(bar_start)

        while pos_ms + loop_ms <= search_end_ms and pos_ms + loop_ms <= total_ms:
            # Map to waveform indices
            i0 = int(n * pos_ms / total_ms)
            i1 = int(n * (pos_ms + loop_ms) / total_ms)
            i_mid = (i0 + i1) // 2
            if i1 - i0 < 4:
                pos_ms += bg.bars_to_ms(1)
                continue

            # Check energy
            heights = [p.height for p in track.waveform[i0:i1]]
            mean_e = sum(heights) / len(heights)
            if mean_e < min_energy:
                pos_ms += bg.bars_to_ms(1)
                continue

            # Check spectral similarity between halves
            similarity = _spectral_similarity(track.waveform, i0, i_mid, i1)
            if similarity > best_score:
                best_score = similarity
                best_pos = pos_ms

            pos_ms += bg.bars_to_ms(1)

        if best_pos is not None and best_score >= min_similarity:
            return (best_pos, loop_bars, best_score)

            pos_ms += bg.bars_to_ms(1)  # slide by 1 bar

    return None


class CueStrategy:
    """Proposes cue placements based on phrase analysis and the cue system."""

    def __init__(
        self,
        memory_offset_bars: int = 16,
        loop_length_bars: int = 4,
    ) -> None:
        self.memory_offset_bars = memory_offset_bars
        self.loop_length_bars = loop_length_bars

    def propose(self, track: Track) -> CueProposal:
        """Generate a cue proposal for a track based on its phrase structure."""
        bg = track.beat_grid
        phrases = track.phrases
        hot_cues: list[CuePoint] = []
        memory_cues: list[CuePoint] = []
        confidence: dict[str, float] = {}
        notes: list[str] = []

        # Build positions dict keyed by pad letter
        positions: dict[str, float] = {}

        # --- A: First Beat ---
        first_beat_ms = bg.beat_to_ms(1)
        positions["A"] = first_beat_ms
        confidence["A"] = 1.0
        notes.append("A (First Beat): beat 1")

        # --- D: Drop (first Chorus or Up after ~25% of track) ---
        # The Drop is the first major energy peak after the intro section.
        # Data shows it's typically around 30% into the track (median).
        # Look for the first Chorus (or Up preceded by a Chorus) that's
        # at least 25% into the track. Fallback to first Chorus after
        # the first Up→Chorus cycle.
        choruses = [p for p in phrases if p.label == "Chorus"]
        drop_candidates = [p for p in phrases if p.label in ("Chorus", "Up")]
        min_drop_ms = track.duration_ms * 0.20  # at least 20% into track
        if choruses:
            # Primary: first Chorus at or after 20% mark
            late_choruses = [c for c in choruses if c.position_ms >= min_drop_ms]
            if late_choruses:
                drop_phrase = late_choruses[0]
                notes.append(f"D (Drop): first Chorus after 20% at beat {drop_phrase.beat_start}")
            else:
                # All choruses are early — check for an Up after the last early Chorus
                last_early_chorus = choruses[-1]
                ups_after = [p for p in phrases
                             if p.label == "Up"
                             and p.position_ms > last_early_chorus.position_ms]
                if ups_after:
                    drop_phrase = ups_after[0]
                    notes.append(
                        f"D (Drop): Up after early Chorus, beat {drop_phrase.beat_start}"
                    )
                else:
                    # Last resort: last Chorus
                    drop_phrase = choruses[-1]
                    notes.append(f"D (Drop): last Chorus at beat {drop_phrase.beat_start}")
            positions["D"] = drop_phrase.position_ms
            confidence["D"] = 0.85
        else:
            notes.append("D (Drop): no Chorus found — skipped")
            confidence["D"] = 0.0

        # --- B/C: N bars before the Drop ---
        # Steps the offset down (32/16 -> 16/8 -> ... ) if the track's intro
        # is too short to fit the full lead-in before the first beat.
        def bars_before_drop(pad: str, bars: int) -> None:
            if "D" not in positions:
                confidence[pad] = 0.0
                notes.append(f"{pad} ({bars} Bars Before Drop): no Drop to anchor from")
                return
            drop_ms = positions["D"]
            offset = bars
            while offset >= 1 and drop_ms - bg.bars_to_ms(offset) < first_beat_ms:
                offset //= 2
            pos_ms = first_beat_ms if offset < 1 else drop_ms - bg.bars_to_ms(offset)
            positions[pad] = pos_ms
            confidence[pad] = 0.85 if offset == bars else 0.5
            if offset == bars:
                notes.append(f"{pad} ({bars} Bars Before Drop): {bars} bars before Drop")
            else:
                notes.append(f"{pad} ({bars} Bars Before Drop): only {offset} bars before Drop (short intro)")

        bars_before_drop("B", 32)
        bars_before_drop("C", 16)

        # --- E: Breakdown (first Down/Bridge after Drop) ---
        if "D" in positions:
            drop_ms = positions["D"]
            downs_after = [p for p in phrases if p.label in ("Down", "Bridge") and p.position_ms > drop_ms]
            if downs_after:
                breakdown_phrase = downs_after[0]
                positions["E"] = breakdown_phrase.position_ms
                confidence["E"] = 0.85
                notes.append(f"E (Breakdown): {breakdown_phrase.label} at beat {breakdown_phrase.beat_start}")
            else:
                confidence["E"] = 0.0
                notes.append("E (Breakdown): no Down/Bridge found after Drop")
        else:
            downs = [p for p in phrases if p.label in ("Down", "Bridge")]
            if downs:
                positions["E"] = downs[0].position_ms
                confidence["E"] = 0.3
                notes.append(f"E (Breakdown): no Drop, using first Down at beat {downs[0].beat_start}")
            else:
                confidence["E"] = 0.0
                notes.append("E (Breakdown): no Down/Bridge found")

        # --- F: Outro ---
        outros = [p for p in phrases if p.label == "Outro"]
        if outros:
            positions["F"] = outros[0].position_ms
            confidence["F"] = 0.9
            notes.append(f"F (Outro): Outro at beat {outros[0].beat_start}")
        elif phrases:
            positions["F"] = phrases[-1].position_ms
            confidence["F"] = 0.4
            notes.append(f"F (Outro): no Outro found, using last phrase at beat {phrases[-1].beat_start}")
        else:
            confidence["F"] = 0.0
            notes.append("F (Outro): no phrases at all")

        # --- Build CuePoint objects ---
        for slot in CUE_SYSTEM:
            pad = slot.pad
            if pad not in positions:
                continue

            pos_ms = positions[pad]
            loop_end = None
            if slot.is_loop:
                loop_end = pos_ms + bg.bars_to_ms(self.loop_length_bars)

            hot_cues.append(CuePoint(
                kind=slot.kind,
                position_ms=pos_ms,
                loop_end_ms=loop_end,
                color_table_index=slot.hot_cue_color_table_index,
                color=slot.hot_cue_color,
                comment=slot.hot_cue_label,
            ))

            # Memory cue
            if slot.memory_offset_bars == 0:
                mem_pos = pos_ms
            else:
                # Prefer the full offset; if the event is too early, step down
                # by halves (16 -> 8 -> 4 bars) so the warning stays phrase-aligned.
                first_beat_ms = bg.beat_to_ms(1)
                offset = self.memory_offset_bars
                while offset >= 1 and pos_ms - bg.bars_to_ms(offset) < first_beat_ms:
                    offset //= 2
                if offset < 1:
                    mem_pos = first_beat_ms
                else:
                    mem_pos = pos_ms - bg.bars_to_ms(offset)
                    # Snap to nearest downbeat (bar start)
                    mem_beat = bg.ms_to_beat(mem_pos)
                    bar_beat = ((mem_beat - 1) // 4) * 4 + 1
                    mem_pos = bg.beat_to_ms(bar_beat)
                    if offset != self.memory_offset_bars:
                        notes.append(f"{pad} memory: only {offset} bars before event")

            mem_loop_end = None
            if slot.is_loop:
                mem_loop_end = mem_pos + bg.bars_to_ms(self.loop_length_bars)

            memory_cues.append(CuePoint(
                kind=0,
                position_ms=mem_pos,
                loop_end_ms=mem_loop_end,
                color_table_index=slot.memory_cue_color_table_index,
                color=slot.memory_cue_color,
                comment=slot.memory_cue_label,
            ))

        return CueProposal(
            track=track,
            hot_cues=hot_cues,
            memory_cues=memory_cues,
            confidence=confidence,
            notes=notes,
        )
