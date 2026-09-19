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
        min_confidence: float = 0.85,
    ) -> None:
        self.memory_offset_bars = memory_offset_bars
        self.loop_length_bars = loop_length_bars
        self.min_confidence = min_confidence

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

        # --- C: Drop (first Chorus or Up after ~25% of track) ---
        # The Drop is the first major energy peak after the intro section.
        # Data shows it's typically around 30% into the track (median).
        # Look for the first Chorus (or Up preceded by a Chorus) that's
        # at least 25% into the track. Fallback to first Chorus after
        # the first Up→Chorus cycle.
        choruses = [p for p in phrases if p.label == "Chorus"]
        min_drop_ms = track.duration_ms * 0.20  # at least 20% into track
        if choruses:
            # Primary: first Chorus at or after 20% mark
            late_choruses = [c for c in choruses if c.position_ms >= min_drop_ms]
            if late_choruses:
                drop_phrase = late_choruses[0]
                notes.append(f"C (Drop): first Chorus after 20% at beat {drop_phrase.beat_start}")
            else:
                # All choruses are early — check for an Up after the last early Chorus
                last_early_chorus = choruses[-1]
                ups_after = [p for p in phrases
                             if p.label == "Up"
                             and p.position_ms > last_early_chorus.position_ms]
                if ups_after:
                    drop_phrase = ups_after[0]
                    notes.append(
                        f"C (Drop): Up after early Chorus, beat {drop_phrase.beat_start}"
                    )
                else:
                    # Last resort: last Chorus
                    drop_phrase = choruses[-1]
                    notes.append(f"C (Drop): last Chorus at beat {drop_phrase.beat_start}")
            positions["C"] = drop_phrase.position_ms
            confidence["C"] = 0.85
        else:
            notes.append("C (Drop): no Chorus found — skipped")
            confidence["C"] = 0.0

        # --- B: 16 Bars Before Vocal ---
        # Finds the first strong, sustained vocal onset (via PVDI vocal
        # detection) before the Drop, then backs up 16 bars from it. This is
        # the entry point for bringing in another track's vocal/acapella so
        # it lands right as this track's own vocal would — or for starting
        # this track early enough that its vocal arrives in the pocket.
        # Falls back to a phrase heuristic if there's no vocal data.
        vocal_ms: float | None = None
        vocal_conf = 0.0
        search_end_ms = positions.get("C", track.duration_ms)
        if track.vocal_track:
            frame_ms = 1024 / 22050 * 1000  # ~46.4ms per PVDI frame
            vt = track.vocal_track
            min_frames = int(2000 / frame_ms)  # require at least 2s of vocal
            i = 0
            while i < len(vt):
                if vt[i] >= 3:  # strong vocal confidence
                    start = i
                    while i < len(vt) and vt[i] > 0:
                        i += 1
                    region_ms = start * frame_ms
                    if i - start >= min_frames and region_ms < search_end_ms:
                        # Snap to nearest phrase boundary
                        best_phrase = None
                        best_dist = float("inf")
                        for p in phrases:
                            dist = abs(p.position_ms - region_ms)
                            if dist < best_dist:
                                best_dist = dist
                                best_phrase = p
                        if best_phrase and best_dist < bg.bars_to_ms(4):
                            vocal_ms = best_phrase.position_ms
                            vocal_conf = 0.85
                            notes.append(
                                f"B: vocal onset at {region_ms / 1000:.1f}s, "
                                f"snapped to {best_phrase.label} beat {best_phrase.beat_start}"
                            )
                        else:
                            snap_beat = bg.ms_to_beat(region_ms)
                            bar_beat = ((snap_beat - 1) // 4) * 4 + 1
                            vocal_ms = bg.beat_to_ms(bar_beat)
                            vocal_conf = 0.8
                            notes.append(f"B: vocal onset at {region_ms / 1000:.1f}s, snapped to beat {bar_beat}")
                        break
                else:
                    i += 1

        if vocal_ms is None:
            ups_before = [
                p for p in phrases
                if p.label in ("Up", "Verse1", "Verse2", "Verse3", "Verse4", "Verse5", "Verse6")
                and p.position_ms < search_end_ms
            ]
            if ups_before:
                vocal_ms = ups_before[-1].position_ms
                vocal_conf = 0.5
                notes.append(f"B: no vocal data, using {ups_before[-1].label} at beat {ups_before[-1].beat_start}")
            else:
                before_drop = [p for p in phrases if p.position_ms < search_end_ms]
                if before_drop:
                    vocal_ms = before_drop[-1].position_ms
                    vocal_conf = 0.3
                    notes.append(f"B: no vocal data, fallback to {before_drop[-1].label} at beat {before_drop[-1].beat_start}")
                else:
                    notes.append("B: no vocal data and no phrase to anchor from")

        if vocal_ms is not None:
            offset = 16
            while offset >= 1 and vocal_ms - bg.bars_to_ms(offset) < first_beat_ms:
                offset //= 2
            pos_ms = first_beat_ms if offset < 1 else vocal_ms - bg.bars_to_ms(offset)
            positions["B"] = pos_ms
            confidence["B"] = vocal_conf if offset == 16 else min(vocal_conf, 0.5)
            if offset != 16:
                notes.append(f"B: only {offset} bars before vocal (short lead-in)")
        else:
            confidence["B"] = 0.0

        # --- D: Breakdown (first Down/Bridge after Drop) ---
        if "C" in positions:
            drop_ms = positions["C"]
            downs_after = [p for p in phrases if p.label in ("Down", "Bridge") and p.position_ms > drop_ms]
            if downs_after:
                breakdown_phrase = downs_after[0]
                positions["D"] = breakdown_phrase.position_ms
                confidence["D"] = 0.85
                notes.append(f"D (Breakdown): {breakdown_phrase.label} at beat {breakdown_phrase.beat_start}")
            else:
                confidence["D"] = 0.0
                notes.append("D (Breakdown): no Down/Bridge found after Drop")
        else:
            downs = [p for p in phrases if p.label in ("Down", "Bridge")]
            if downs:
                positions["D"] = downs[0].position_ms
                confidence["D"] = 0.3
                notes.append(f"D (Breakdown): no Drop, using first Down at beat {downs[0].beat_start}")
            else:
                confidence["D"] = 0.0
                notes.append("D (Breakdown): no Down/Bridge found")

        # --- E: Outro ---
        outros = [p for p in phrases if p.label == "Outro"]
        if outros:
            positions["E"] = outros[0].position_ms
            confidence["E"] = 0.9
            notes.append(f"E (Outro): Outro at beat {outros[0].beat_start}")
        elif phrases:
            positions["E"] = phrases[-1].position_ms
            confidence["E"] = 0.4
            notes.append(f"E (Outro): no Outro found, using last phrase at beat {phrases[-1].beat_start}")
        else:
            confidence["E"] = 0.0
            notes.append("E (Outro): no phrases at all")

        # --- Build CuePoint objects ---
        for slot in CUE_SYSTEM:
            pad = slot.pad
            if pad not in positions:
                continue
            if confidence.get(pad, 0.0) < self.min_confidence:
                notes.append(f"{pad}: confidence {confidence.get(pad, 0.0):.2f} below threshold — skipped")
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
