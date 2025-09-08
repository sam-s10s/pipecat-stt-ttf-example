#!/usr/bin/env python3
"""
Script to plot audio data from JSONL files.

Usage:
    python plot_results.py <folder_path>

Example:
    python plot_results.py output/test
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_audio_data(jsonl_path: str) -> Tuple[List[datetime], np.ndarray]:
    """
    Load audio data from a JSONL file.

    Args:
        jsonl_path: Path to the audio.jsonl file

    Returns:
        Tuple of (timestamps, audio_samples)
    """
    timestamps = []
    audio_chunks = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Parse timestamp
            ts = datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
            timestamps.append(ts)

            # Get audio payload
            audio_chunks.append(np.array(data["payload"], dtype=np.int16))

    # Concatenate all audio chunks
    audio_samples = np.concatenate(audio_chunks)

    return timestamps, audio_samples


def load_stt_timestamps(jsonl_path: str) -> Tuple[List[datetime], List[bool]]:
    """
    Load timestamps from an STT JSONL file, filtering out empty transcripts.

    Args:
        jsonl_path: Path to the STT .jsonl file

    Returns:
        Tuple of (timestamps, is_final_flags) for entries with non-empty transcripts
    """
    timestamps = []
    is_final_flags = []

    if not os.path.exists(jsonl_path):
        return timestamps, is_final_flags

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Check if transcript is not empty
            text = data.get("payload", {}).get("text", "")

            if text.strip():  # Only include non-empty transcripts
                # Parse timestamp
                ts = datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
                timestamps.append(ts)

                # Check if it's final
                is_final = data.get("payload", {}).get("final", False)
                is_final_flags.append(is_final)

    return timestamps, is_final_flags


def find_stt_files(folder_path: Path) -> List[str]:
    """
    Find all STT provider JSONL files (files starting with stt_).

    Args:
        folder_path: Path to the folder containing JSONL files

    Returns:
        List of STT provider names (filenames without .jsonl extension)
    """
    stt_files = []

    for file_path in folder_path.glob("stt_*.jsonl"):
        stt_files.append(file_path.stem)  # filename without extension

    return sorted(stt_files)


def load_speaking_events(jsonl_path: str) -> List[Tuple[datetime, bool]]:
    """
    Load speaking start/stop events from speaking.jsonl file.

    Args:
        jsonl_path: Path to the speaking.jsonl file

    Returns:
        List of (timestamp, is_speaking) tuples
    """
    speaking_events = []

    if not os.path.exists(jsonl_path):
        return speaking_events

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Parse timestamp and speaking state
            ts = datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
            is_speaking = data.get("payload", {}).get("speaking", False)

            speaking_events.append((ts, is_speaking))

    return speaking_events


def add_vad_shading(
    ax,
    speaking_events: List[Tuple[datetime, bool]],
    first_audio_time: datetime,
    duration_seconds: float,
):
    """
    Add light blue background shading for VAD (speaking) periods.

    Args:
        ax: Matplotlib axis to add shading to
        speaking_events: List of (timestamp, is_speaking) tuples
        first_audio_time: First audio timestamp for offset calculation
        duration_seconds: Total duration of audio
    """
    if not speaking_events:
        return

    speaking_start = None

    for speaking_timestamp, is_speaking in speaking_events:
        time_offset = (speaking_timestamp - first_audio_time).total_seconds()

        # Only process events within audio duration
        if 0 <= time_offset <= duration_seconds:
            if is_speaking and speaking_start is None:
                # Start of speaking period
                speaking_start = time_offset
            elif not is_speaking and speaking_start is not None:
                # End of speaking period - add shading
                ax.axvspan(speaking_start, time_offset, alpha=0.2, color="lightblue", zorder=0)
                speaking_start = None

    # Handle case where speaking continues to end of audio
    if speaking_start is not None:
        ax.axvspan(speaking_start, duration_seconds, alpha=0.2, color="lightblue", zorder=0)


def plot_audio_with_stt_providers(
    timestamps: List[datetime],
    audio_samples: np.ndarray,
    output_folder: str,
    stt_data: dict,
    speaking_events: List[Tuple[datetime, bool]] = None,
    sample_rate: int = 16000,
):
    """
    Plot the audio waveform with STT provider timestamp markers.

    Args:
        timestamps: List of timestamps for each audio chunk
        audio_samples: Concatenated audio samples
        output_folder: Folder name for the plot title
        stt_data: Dict mapping provider names to (timestamps, is_final_flags) tuples
        speaking_events: List of (timestamp, is_speaking) tuples for VAD shading
        sample_rate: Audio sample rate (default 16000 Hz)
    """
    # Create time axis in seconds
    duration_seconds = len(audio_samples) / sample_rate
    time_axis = np.linspace(0, duration_seconds, len(audio_samples))

    # Convert to float and normalise to [-1, 1] range
    audio_float = audio_samples.astype(np.float32) / 32768.0

    # Calculate number of subplots needed (1 main + 1 per STT provider)
    num_stt_providers = len(stt_data)
    total_subplots = 1 + num_stt_providers

    # Create height ratios (main plot gets more space)
    height_ratios = [2] + [1] * num_stt_providers

    # Create the plot with subplots
    fig, axes = plt.subplots(
        total_subplots, 1, figsize=(15, 4 + 2 * num_stt_providers), height_ratios=height_ratios
    )

    # Handle single subplot case
    if total_subplots == 1:
        axes = [axes]

    # Main waveform plot
    ax_main = axes[0]

    # Calculate relative time offset from first audio timestamp
    first_audio_time = timestamps[0] if timestamps else None

    # Add VAD shading first (behind everything)
    if speaking_events and first_audio_time:
        add_vad_shading(ax_main, speaking_events, first_audio_time, duration_seconds)

    # Add audio waveform on top of shading
    ax_main.plot(time_axis, audio_float, linewidth=0.5, color="blue", alpha=0.7)
    ax_main.set_title(f"STT Comparison - {output_folder}", fontsize=14, fontweight="bold")
    ax_main.set_ylabel("Audio")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, duration_seconds)
    ax_main.set_ylim(-0.5, 0.5)
    ax_main.set_yticklabels([])  # Remove y-axis tick labels

    # Set custom grid: vertical lines every second, labels every 5 seconds
    ax_main.xaxis.set_major_locator(ticker.MultipleLocator(5))  # Major ticks every 5 seconds
    ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor ticks every 1 second
    ax_main.grid(True, which="major", alpha=0.3)
    ax_main.grid(True, which="minor", alpha=0.1, linestyle=":")

    # Add speaking events to main chart
    speaking_start_plotted = False
    speaking_stop_plotted = False
    if speaking_events and first_audio_time:
        for speaking_timestamp, is_speaking in speaking_events:
            # Convert timestamp difference to seconds
            time_offset = (speaking_timestamp - first_audio_time).total_seconds()

            # Only plot if within the audio duration
            if 0 <= time_offset <= duration_seconds:
                # Use green for speaking start, orange for speaking stop
                color = "green" if is_speaking else "orange"
                label = None

                # Add labels for legend (only once per type)
                if is_speaking and not speaking_start_plotted:
                    label = "Speaking Start"
                    speaking_start_plotted = True
                elif not is_speaking and not speaking_stop_plotted:
                    label = "Speaking Stop"
                    speaking_stop_plotted = True

                ax_main.axvline(x=time_offset, color=color, alpha=0.8, linewidth=1.5, label=label)

    # Add legend to main chart including VAD shading
    if speaking_events:
        from matplotlib.patches import Patch

        legend_elements = []

        # Add VAD shading patch
        vad_patch = Patch(color="lightblue", alpha=0.2, label="VAD Speaking Periods")
        legend_elements.append(vad_patch)

        # Add existing line elements
        handles, labels = ax_main.get_legend_handles_labels()
        legend_elements.extend(handles)

        ax_main.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Create subplot for each STT provider
    for i, (provider_name, (stt_timestamps, is_final_flags)) in enumerate(stt_data.items()):
        ax = axes[i + 1]

        # Add VAD shading first (behind everything)
        if speaking_events and first_audio_time:
            add_vad_shading(ax, speaking_events, first_audio_time, duration_seconds)

        # Add darker waveform background
        ax.plot(time_axis, audio_float, linewidth=0.4, color="gray", alpha=0.7)

        # Set up the subplot
        ax.set_xlim(0, duration_seconds)
        ax.set_ylim(-0.5, 0.5)  # Slightly larger than audio range for visibility

        # Remove 'stt_' prefix from display name and make uppercase
        display_name = provider_name.replace("stt_", "").upper()
        ax.set_ylabel(display_name)
        ax.set_yticklabels([])  # Remove y-axis tick labels

        # Set custom grid: vertical lines every second, labels every 5 seconds
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))  # Major ticks every 5 seconds
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor ticks every 1 second
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1, linestyle=":")

        # Add vertical lines for STT timestamps
        interim_plotted = False
        final_plotted = False
        if stt_timestamps and first_audio_time:
            for stt_timestamp, is_final in zip(stt_timestamps, is_final_flags):
                # Convert timestamp difference to seconds
                time_offset = (stt_timestamp - first_audio_time).total_seconds()

                # Only plot if within the audio duration
                if 0 <= time_offset <= duration_seconds:
                    # Use purple for final transcripts, red for interim
                    color = "purple" if is_final else "red"
                    label = None

                    # Add labels for legend (only once per type)
                    if is_final and not final_plotted:
                        label = "Final Transcript"
                        final_plotted = True
                    elif not is_final and not interim_plotted:
                        label = "Interim Transcript"
                        interim_plotted = True

                    ax.axvline(x=time_offset, color=color, alpha=0.7, linewidth=2, label=label)

        # Add legend to STT provider chart
        if stt_timestamps:
            ax.legend(loc="upper right", fontsize=8)

    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {args.output}")

    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot audio data from JSONL files")
    parser.add_argument("folder", help="Folder containing audio.jsonl file (e.g., output/test)")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test-result.png",
        help="Output filename (default: test-result.png)",
    )

    args = parser.parse_args()

    # Construct path to audio.jsonl
    folder_path = Path(args.folder)
    jsonl_path = folder_path / "audio.jsonl"

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist!")
        return 1

    print(f"Loading audio data from: {jsonl_path}")

    try:
        timestamps, audio_samples = load_audio_data(str(jsonl_path))

        print(f"Loaded {len(audio_samples):,} audio samples from {len(timestamps)} chunks")

        # Find all STT provider files
        stt_providers = find_stt_files(folder_path)

        if not stt_providers:
            print("No STT provider files found (looking for stt_*.jsonl files)")
            return 1

        print(f"Found STT providers: {', '.join(stt_providers)}")

        # Load data for each STT provider
        stt_data = {}
        for provider in stt_providers:
            provider_path = folder_path / f"{provider}.jsonl"
            provider_data = load_stt_timestamps(str(provider_path))

            if provider_data[0]:  # Check if timestamps list is not empty
                stt_data[provider] = provider_data
                print(f"Loaded {len(provider_data[0])} {provider} timestamps with transcripts")
            else:
                print(f"No {provider} timestamps with transcripts found")

        if not stt_data:
            print("No STT data with transcripts found")
            return 1

        # Load speaking events
        speaking_path = folder_path / "speaking.jsonl"
        speaking_events = load_speaking_events(str(speaking_path))

        if speaking_events:
            print(f"Loaded {len(speaking_events)} speaking events")
        else:
            print("No speaking events found")

        # Plot waveform with all STT providers and speaking events
        plot_audio_with_stt_providers(
            timestamps,
            audio_samples,
            args.folder,
            stt_data,
            speaking_events,
            args.sample_rate,
            args,
        )

        print("Plotting complete!")
        return 0

    except Exception as e:
        print(f"Error processing audio data: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
