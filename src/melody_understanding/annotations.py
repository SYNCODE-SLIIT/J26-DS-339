# Turns MIDI events into a list of notes
"""Read MIDI annotations independently of the exploratory notebook parser."""
from collections import defaultdict, deque
from pathlib import Path
import mido

# Open the MIDI File and Prepare to Read Notes
# Input : Midi File location , Output : A sorted list of notes, Counts of potential annotation problems
def read_midi_notes(path):
    """Return sorted (start seconds, end seconds, MIDI pitch) and anomalies.
    Mido merges synchronous tracks and applies tempo changes when iterating.
    Type 2 contains independent sequences and cannot use a shared time axis.
    """
    midi = mido.MidiFile(Path(path))
    
    # Reject tracks that can't be interpreted on a shared musical timeline
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("Independent tracks/SMPTE timing are unsupported")
    
    # active remembers notes that have started and are waiting to end
    active = defaultdict(deque)
    notes, seconds, unmatched = [], 0.0, 0
    
    # Reading Events and Matching Note Starts to Endings
    for event in midi:
        seconds += event.time
        if event.type == "note_on" and event.velocity > 0:
            active[(event.channel, event.note)].append(seconds)
        elif event.type == "note_off" or (event.type == "note_on" and event.velocity == 0):
            key = (event.channel, event.note)
            if active[key]:
                notes.append((active[key].popleft(), seconds, event.note))
            else:
                unmatched += 1

    # Return Notes and Report Problems
    return sorted(notes), {
        "unmatched_note_offs": unmatched,
        "unclosed_note_ons": sum(len(v) for v in active.values()),
        "nonpositive_notes": sum(b <= a for a, b, _ in notes),
    }

# counts notes starting during an earlier note
def overlap_count(notes):
    """Count events beginning before the preceding active note has ended."""
    end, count = -1.0, 0
    for start, stop, _ in sorted(notes):
        count += int(start < end - 1e-7)
        end = max(end, stop)
    return count
