"""Elf ECG for lab 12: generator, scores, player and plots.

An elf ECG is a synthetic ECG whose QRS complex has the shape of a Christmas
tree (the star is the R peak), whose RR intervals follow the rhythm of the
chorus of Jingle Bells and whose tree heights encode the pitch of the notes.

Everything is counted in samples; FS is used only for the time axis and the
audio. Reproducible: every random draw comes from the generator `rng` (or the
`seed`) given by the caller.

For the notebook: N, FS, elf_ecg, make_dataset, score, score_many,
score_title, play, plot_ecg. Names with a leading underscore are internals.

    python elf.py            # self-check: scores of real and sick records
"""
import numpy as np
from scipy.signal import find_peaks

FS = 250          # sampling frequency [Hz]
N = 2048          # samples per record (8.2 s)
W = 45            # tree width in samples (180 ms, a wide "elf" QRS); odd, so the star sits on a sample
SEMI = 0.08       # relative change of the tree height per semitone
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DURATIONS = [1, 1, 2, 1, 1, 2, 1, 1, 1.5, 0.5, 2]              # beats between consecutive notes (Jingle Bells chorus)
PITCHES = [64, 64, 64, 64, 64, 64, 64, 67, 60, 62, 64, 64]      # MIDI; one more note than durations: the closing beat
BEAT_RANGE = (96, 124)                                          # samples per beat (tempo range); shortest RR >= 48 > W
REF = 64                                                        # the median note: height 1, decoded as E4

# Tree silhouette on W samples: the star (R) is 1 at the centre, the snow (Q, S) is -0.15.
_left = [(0.00, 0.00), (0.03, -0.15), (0.10, -0.15),            # Q: snow under the tree
         (0.11, 0.38), (0.23, 0.52), (0.24, 0.30),              # tier 1 tip, deep notch
         (0.25, 0.30), (0.34, 0.70), (0.35, 0.50),              # tier 2 tip, notch
         (0.36, 0.50), (0.44, 0.84), (0.455, 0.76),             # tier 3 tip, notch
         (0.475, 0.80), (0.50, 1.00)]                           # top and the star spike
_pts = np.array(_left + [(1 - t, h) for t, h in reversed(_left[:-1])])
TREE = np.interp(np.linspace(0, 1, W), _pts[:, 0], _pts[:, 1])


def _gauss(i, mu, s, a):
    return a * np.exp(-0.5 * ((i - mu) / s) ** 2)


def elf_ecg(rng, sick=False):
    """One record of N samples (float32, mV).

    sick=True drops one beat (the elf skips a note): a negative example for the scores.
    """
    beat = rng.uniform(*BEAT_RANGE)                                   # tempo: samples per beat
    rr = np.array(DURATIONS, float) * beat * rng.normal(1, 0.015, len(DURATIONS))   # 1.5 % jitter
    start = rng.uniform(40, N - rr.sum() - 40)
    r = start + np.concatenate([[0], np.cumsum(rr)])
    pitch = list(PITCHES)
    if sick:
        k = rng.integers(1, len(r) - 1)
        r = np.delete(r, k)
        pitch.pop(k)
    i = np.arange(N, dtype=float)
    gain = rng.uniform(0.8, 1.2)
    x = np.zeros(N)
    gap = np.diff(np.concatenate([[r[0] - 200], r, [r[-1] + 200]]))
    for k, (rk, m) in enumerate(zip(r, pitch)):
        h = gain * (1 + SEMI * (m - REF)) * rng.normal(1, 0.005)
        room_p = np.clip((gap[k] - 60) / 40, 0, 1)                    # P and T fade when the RR is short
        room_t = np.clip((gap[k + 1] - 60) / 40, 0, 1)
        x += _gauss(i, rk - 32, 4, 0.10 * gain * room_p)              # P wave
        x += _gauss(i, rk + 36, 6, 0.15 * gain * room_t)              # T wave
        a = int(round(rk)) - W // 2
        x[a:a + W] += h * TREE
    x += 0.05 * np.sin(2 * np.pi * i / rng.uniform(800, 2000) + rng.uniform(0, 6.3))   # baseline wander
    x += rng.normal(0, 0.01, N)
    return x.astype(np.float32)


def make_dataset(n, seed=0):
    """n records as a float32 array (n, N)."""
    rng = np.random.default_rng(seed)
    return np.stack([elf_ecg(rng) for _ in range(n)])


# ---------------------------------------------------------------- analysis
def _detect_r(x):
    """Simple R-peak detector: prominent maxima at least 30 samples apart."""
    x = np.asarray(x, float)
    pk, _ = find_peaks(x, prominence=0.45 * (np.percentile(x, 99.5) - np.median(x)), distance=30)
    return pk[(pk >= W // 2) & (pk < len(x) - W // 2)]       # a whole tree must fit into the record


def _tree_corr(x, pk):
    """Pearson correlation of every detected beat with the tree template."""
    x = np.asarray(x, float)
    return np.array([np.corrcoef(x[p - W // 2:p + W // 2 + 1], TREE)[0, 1] for p in pk])


def _rhythm_error(pk):
    """Tempo-invariant rhythm error: mean |log(RR / (fitted tempo * duration))|; inf for a wrong beat count."""
    d = np.array(DURATIONS, float)
    rr = np.diff(pk).astype(float)
    if len(rr) != len(d):
        return np.inf
    tempo = np.exp(np.mean(np.log(rr / d)))
    return float(np.mean(np.abs(np.log(rr / (tempo * d)))))


def _decode_pitch(x, pk):
    """Tree heights -> MIDI pitches: semitones around the median height, which is anchored at REF."""
    x = np.asarray(x, float)
    if len(pk) == 0:
        return np.zeros(0, int)
    snow = np.array([np.mean(x[p + W // 2 - 4:p + W // 2]) for p in pk])   # S plateau right of the tree
    h = x[pk] - snow                                                       # star minus snow (R - S)
    with np.errstate(all="ignore"):                                        # a flat or pure-noise signal
        rel = np.nan_to_num(h / np.median(h) - 1, posinf=0, neginf=0)
    return REF + np.round(np.clip(rel, -3, 3) / SEMI).astype(int)


def _song_prefix(x, pk):
    """Partial score: how many notes from the start of the record follow the song.

    The first note counts when its pitch is right; every further note counts when
    its pitch and the RR interval before it are right (within 15 % of the tempo,
    which is fitted as the median over the intervals; 15 % is half the distance
    between a note of 1.5 and of 2 beats). 0 ... number of notes. Stricter about
    a single interval than `_rhythm_error`, which averages over the record.
    """
    if len(pk) < 2:
        return 0
    notes = _decode_pitch(x, pk)
    rr = np.diff(pk).astype(float)
    m = min(len(rr), len(DURATIONS))
    tempo = np.exp(np.median(np.log(rr[:m] / np.array(DURATIONS[:m]))))
    k = 0
    while k < min(len(pk), len(PITCHES)) and notes[k] == PITCHES[k] and (
            k == 0 or abs(np.log(rr[k - 1] / (tempo * DURATIONS[k - 1]))) < 0.15):
        k += 1
    return k


def score(x):
    """Scores of one record.

    n_beats  number of detected beats
    tree     median correlation of the beats with the tree template
    trees_ok every beat has a tree correlation r > 0.95
    rhythm   tempo-fitted RR error (inf when the number of beats is wrong)
    melody   fraction of notes with the right pitch (0 when the number of beats is wrong)
    notes    partial score: length of the correct prefix of the song (see _song_prefix)
    valid    trees_ok, the right number of beats with rhythm < 0.08, melody >= 0.8
    """
    pk = _detect_r(x)
    tc = _tree_corr(x, pk)
    re = _rhythm_error(pk)
    mel = float(np.mean(_decode_pitch(x, pk) == np.array(PITCHES))) if len(pk) == len(PITCHES) else 0.0
    trees_ok = bool(len(tc) and tc.min() > 0.95)
    return dict(n_beats=len(pk), tree=float(np.median(tc)) if len(tc) else 0.0, trees_ok=trees_ok,
                rhythm=re, melody=mel, notes=_song_prefix(x, pk),
                valid=trees_ok and re < 0.08 and mel >= 0.8)


def score_many(X):
    """Summary over many records (rows of X); percentages of records unless stated otherwise."""
    S = [score(x) for x in X]
    n = len(PITCHES)
    tempo = [np.median(np.diff(_detect_r(x))) for x, s in zip(X, S) if s["n_beats"] > 2]
    out = {
        "valid [%]": 100 * np.mean([s["valid"] for s in S]),
        "all trees r > 0.95 [%]": 100 * np.mean([s["trees_ok"] for s in S]),
        f"{n} beats [%]": 100 * np.mean([s["n_beats"] == n for s in S]),
        "rhythm < 0.08 [%]": 100 * np.mean([s["rhythm"] < 0.08 for s in S]),
        "melody >= 0.8 [%]": 100 * np.mean([s["melody"] >= 0.8 for s in S]),
        f"mean correct notes from the start (of {n})": np.mean([s["notes"] for s in S]),
        "tempo spread: std of the median RR [samples]": np.std(tempo) if tempo else 0.0,
    }
    return {k: round(float(v), 2) for k, v in out.items()}


def score_title(x):
    """A one-line summary of the scores of a record, for plot titles."""
    s = score(x)
    return (f"beats {s['n_beats']}/{len(PITCHES)}, tree r {s['tree']:.3f}, rhythm err {s['rhythm']:.3f}, "
            f"melody {s['melody']:.2f}, notes from the start {s['notes']}, valid {s['valid']}")


# ---------------------------------------------------------------- player
def play(x, melody=True, sr=8000):
    """ECG monitor as a sound, as an IPython Audio widget: one beep per detected R peak.

    melody=True reads the pitch of every beep from the tree height (the median
    height plays the MIDI note REF), melody=False plays the same beep for all.
    """
    from IPython.display import Audio
    pk = _detect_r(x)
    notes = _decode_pitch(x, pk) if melody else [81] * len(pk)
    audio = np.zeros(int((len(x) / FS + 0.3) * sr))
    tt = np.arange(int(0.18 * sr)) / sr
    for p, m in zip(pk, notes):
        f = 440 * 2 ** ((np.clip(m, 36, 96) - 69) / 12)
        s = int(p / FS * sr)
        tone = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)
        audio[s:s + len(tt)] += tone * np.exp(-tt * 12)
    return Audio(audio / (np.abs(audio).max() + 1e-9) * 0.8, rate=sr, normalize=False)


# ---------------------------------------------------------------- plots
def plot_ecg(ax, x, title=None, zoom_ax=None):
    """One record: the signal, every detected beat filled as a tree with a star and the decoded note.

    zoom_ax: an optional second axis that shows the second beat in detail.
    """
    x = np.asarray(x, float)
    t = np.arange(len(x)) / FS
    pk = _detect_r(x)
    ax.plot(t, x, color="#b0302a", lw=0.9)
    for p in pk:
        seg = slice(p - W // 2, p + W // 2 + 1)
        ax.fill_between(t[seg], x[seg].min(), x[seg], color="#2e7d32", alpha=0.35, lw=0)
    ax.plot(t[pk], x[pk] + 0.07, "*", color="#f4b400", ms=11, mec="#8a6d00", mew=0.5)
    for p, m in zip(pk, _decode_pitch(x, pk)):
        ax.text(t[p], x[p] + 0.2, NOTE_NAMES[m % 12], ha="center", fontsize=8, color="#555")
    ax.set_ylim(x.min() - 0.1, x.max() + 0.4)
    ax.set_xlim(0, len(x) / FS)
    ax.set_ylabel("mV")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    if zoom_ax is not None and len(pk) > 1:
        p = pk[1]
        a, b = max(p - 40, 0), min(p + 50, len(x))
        ms = (t - t[p]) * 1000
        zoom_ax.plot(ms[a:b], x[a:b], color="#b0302a", lw=1.2)
        seg = slice(p - W // 2, p + W // 2 + 1)
        zoom_ax.fill_between(ms[seg], x[seg].min(), x[seg], color="#2e7d32", alpha=0.35, lw=0)
        zoom_ax.plot(0, x[p] + 0.06, "*", color="#f4b400", ms=12, mec="#8a6d00", mew=0.5)
        zoom_ax.set_xticks([-100, 0, 100])
        zoom_ax.set_yticks([])
        zoom_ax.set_title("zoom: one beat [ms]", fontsize=9)
        ax.axvspan(t[a], t[b - 1], color="0.85", zorder=0)


if __name__ == "__main__":
    print("real records:", score_many(make_dataset(300, 1)))
    rng = np.random.default_rng(2)
    sick = np.mean([score(elf_ecg(rng, sick=True))["valid"] for _ in range(300)])
    print(f"sick elf valid {100 * sick:.1f} %")
