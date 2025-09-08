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
    Find all STT provider JSONL files (excluding audio.jsonl).
    
    Args:
        folder_path: Path to the folder containing JSONL files
        
    Returns:
        List of STT provider names (filenames without .jsonl extension)
    """
    stt_files = []
    
    for file_path in folder_path.glob("*.jsonl"):
        if file_path.name != "audio.jsonl":
            stt_files.append(file_path.stem)  # filename without extension
    
    return sorted(stt_files)


def plot_audio_with_stt_providers(
    timestamps: List[datetime],
    audio_samples: np.ndarray,
    output_folder: str,
    stt_data: dict,
    sample_rate: int = 16000,
):
    """
    Plot the audio waveform with STT provider timestamp markers.

    Args:
        timestamps: List of timestamps for each audio chunk
        audio_samples: Concatenated audio samples
        output_folder: Folder name for the plot title
        stt_data: Dict mapping provider names to (timestamps, is_final_flags) tuples
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
    height_ratios = [3] + [1] * num_stt_providers
    
    # Create the plot with subplots
    fig, axes = plt.subplots(total_subplots, 1, figsize=(15, 4 + 2 * num_stt_providers), 
                            height_ratios=height_ratios)
    
    # Handle single subplot case
    if total_subplots == 1:
        axes = [axes]
    
    # Main waveform plot
    ax_main = axes[0]
    ax_main.plot(time_axis, audio_float, linewidth=0.5, color="blue", alpha=0.7)
    ax_main.set_title(f"Audio Waveform - {output_folder}", fontsize=14, fontweight="bold")
    ax_main.set_ylabel("Amplitude")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, duration_seconds)
    
    # Set custom grid: vertical lines every second, labels every 5 seconds
    import matplotlib.ticker as ticker
    ax_main.xaxis.set_major_locator(ticker.MultipleLocator(5))  # Major ticks every 5 seconds
    ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor ticks every 1 second
    ax_main.grid(True, which='major', alpha=0.3)
    ax_main.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Calculate relative time offset from first audio timestamp
    first_audio_time = timestamps[0] if timestamps else None
    
    # Create subplot for each STT provider
    for i, (provider_name, (stt_timestamps, is_final_flags)) in enumerate(stt_data.items()):
        ax = axes[i + 1]
        
        # Add light gray waveform background
        ax.plot(time_axis, audio_float, linewidth=0.3, color="lightgray", alpha=0.5)
        
        # Set up the subplot
        ax.set_xlim(0, duration_seconds)
        ax.set_ylim(-1.2, 1.2)  # Slightly larger than audio range for visibility
        ax.set_ylabel(f"{provider_name.title()}\nTranscripts")
        
        # Set custom grid: vertical lines every second, labels every 5 seconds
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))  # Major ticks every 5 seconds
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor ticks every 1 second
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        # Add vertical lines for STT timestamps
        if stt_timestamps and first_audio_time:
            for stt_timestamp, is_final in zip(stt_timestamps, is_final_flags):
                # Convert timestamp difference to seconds
                time_offset = (stt_timestamp - first_audio_time).total_seconds()
                
                # Only plot if within the audio duration
                if 0 <= time_offset <= duration_seconds:
                    # Use purple for final transcripts, red for interim
                    color = "purple" if is_final else "red"
                    ax.axvline(x=time_offset, color=color, alpha=0.7, linewidth=2)
    
    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel("Time (seconds)")
    
    plt.tight_layout()

    # Save the plot
    plot_filename = f"audio_plot_{output_folder.replace('/', '_')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {plot_filename}")

    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot audio data from JSONL files")
    parser.add_argument("folder", help="Folder containing audio.jsonl file (e.g., output/test)")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)"
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
            print("No STT provider files found (looking for *.jsonl files other than audio.jsonl)")
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
        
        # Plot waveform with all STT providers
        plot_audio_with_stt_providers(timestamps, audio_samples, args.folder, stt_data, args.sample_rate)

        print("Plotting complete!")
        return 0

    except Exception as e:
        print(f"Error processing audio data: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
