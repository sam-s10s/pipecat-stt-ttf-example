#!/usr/bin/env python3
"""
Script to create interactive HTML charts from JSONL files.

Usage:
    python plot-results-html.py <folder_path>

Example:
    python plot-results-html.py output/test
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_audio_data(jsonl_path: str) -> Tuple[List[datetime], List[int]]:
    """
    Load audio data from a JSONL file.

    Args:
        jsonl_path: Path to the audio.jsonl file

    Returns:
        Tuple of (timestamps, audio_samples)
    """
    timestamps = []
    audio_samples = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Parse timestamp
            ts = datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
            timestamps.append(ts)

            # Get audio payload and flatten to list
            audio_chunk = data["payload"]
            audio_samples.extend(audio_chunk)

    return timestamps, audio_samples


def load_stt_timestamps(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load timestamps from an STT JSONL file, filtering out empty transcripts.

    Args:
        jsonl_path: Path to the STT .jsonl file

    Returns:
        List of transcript events with timestamp, text, and final flag
    """
    events = []

    if not os.path.exists(jsonl_path):
        return events

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Check if transcript is not empty
            text = data.get("payload", {}).get("text", "")

            if text.strip():  # Only include non-empty transcripts
                # Parse timestamp
                ts = datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
                is_final = data.get("payload", {}).get("final", False)

                events.append({"timestamp": ts, "text": text.strip(), "is_final": is_final})

    return events


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


def load_speaking_events(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load speaking start/stop events from speaking.jsonl file.

    Args:
        jsonl_path: Path to the speaking.jsonl file

    Returns:
        List of speaking events with timestamp and speaking state
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

            speaking_events.append({"timestamp": ts, "is_speaking": is_speaking})

    return speaking_events


def create_audio_base64(audio_samples: List[int], sample_rate: int) -> str:
    """
    Create a base64-encoded WAV audio data URL from the audio samples.

    Args:
        audio_samples: List of audio sample values
        sample_rate: Audio sample rate

    Returns:
        Base64 data URL for the audio
    """
    import base64
    import io
    import struct
    import wave

    # Create WAV in memory
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Convert to bytes
        audio_bytes = struct.pack("<" + "h" * len(audio_samples), *audio_samples)
        wav_file.writeframes(audio_bytes)

    # Get WAV data and encode as base64
    wav_data = wav_buffer.getvalue()
    base64_data = base64.b64encode(wav_data).decode("utf-8")

    return f"data:audio/wav;base64,{base64_data}"


def prepare_chart_data(
    timestamps: List[datetime],
    audio_samples: List[int],
    stt_data: Dict[str, List[Dict[str, Any]]],
    speaking_events: List[Dict[str, Any]],
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """
    Prepare data for Chart.js visualization.
    """
    if not timestamps:
        return {}

    first_audio_time = timestamps[0]
    duration_seconds = len(audio_samples) / sample_rate

    # Downsample audio for visualization (every 100th sample)
    downsample_factor = 50
    downsampled_audio = audio_samples[::downsample_factor]
    downsampled_times = [i * downsample_factor / sample_rate for i in range(len(downsampled_audio))]

    # Normalize audio to [-0.5, 0.5] range for better visibility
    normalized_audio = [sample / 65536.0 for sample in downsampled_audio]

    # Prepare audio chart data
    audio_chart_data = {
        "labels": downsampled_times,
        "datasets": [
            {
                "label": "Audio Waveform",
                "data": normalized_audio,
                "borderColor": "rgb(59, 130, 246)",
                "backgroundColor": "rgba(59, 130, 246, 0.1)",
                "borderWidth": 1,
                "pointRadius": 0,
                "fill": True,
            }
        ],
    }

    # Prepare VAD shading data
    vad_regions = []
    speaking_start = None

    for event in speaking_events:
        time_offset = (event["timestamp"] - first_audio_time).total_seconds()

        if 0 <= time_offset <= duration_seconds:
            if event["is_speaking"] and speaking_start is None:
                speaking_start = time_offset
            elif not event["is_speaking"] and speaking_start is not None:
                vad_regions.append({"start": speaking_start, "end": time_offset})
                speaking_start = None

    # Handle case where speaking continues to end
    if speaking_start is not None:
        vad_regions.append({"start": speaking_start, "end": duration_seconds})

    # Prepare speaking events data
    speaking_events_data = []
    for event in speaking_events:
        time_offset = (event["timestamp"] - first_audio_time).total_seconds()
        if 0 <= time_offset <= duration_seconds:
            speaking_events_data.append({"x": time_offset, "is_speaking": event["is_speaking"]})

    # Prepare STT data
    stt_chart_data = {}
    for provider, events in stt_data.items():
        provider_events = []
        for event in events:
            time_offset = (event["timestamp"] - first_audio_time).total_seconds()
            if 0 <= time_offset <= duration_seconds:
                provider_events.append(
                    {"x": time_offset, "text": event["text"], "is_final": event["is_final"]}
                )

        display_name = provider.replace("stt_", "").upper()
        stt_chart_data[display_name] = provider_events

    return {
        "audio": audio_chart_data,
        "vad_regions": vad_regions,
        "speaking_events": speaking_events_data,
        "stt_data": stt_chart_data,
        "duration": duration_seconds,
    }


def create_html_page(
    chart_data: Dict[str, Any],
    output_folder: str,
    html_folder: Path,
    audio_file: str,
    output_filename: str,
):
    """
    Create the interactive HTML page with Chart.js.
    """
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STT Comparison - {output_folder}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
</head>
<body class="bg-gray-100 p-6">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold text-gray-900 mb-6">STT Comparison - {output_folder}</h1>
        
        <!-- Compact Legend -->
        <div class="bg-white rounded-lg shadow-md p-3 mb-4">
            <div class="flex flex-wrap items-center gap-4 text-xs">
                <div class="flex items-center">
                    <div class="w-3 h-3 bg-blue-500 opacity-20 mr-1"></div>
                    <span>VAD</span>
                </div>
                <div class="flex items-center">
                    <div class="w-3 h-0.5 bg-green-500 mr-1"></div>
                    <span>Start</span>
                </div>
                <div class="flex items-center">
                    <div class="w-3 h-0.5 bg-orange-500 mr-1"></div>
                    <span>Stop</span>
                </div>
                <div class="flex items-center">
                    <div class="w-3 h-0.5 bg-red-500 mr-1"></div>
                    <span>Interim</span>
                </div>
                <div class="flex items-center">
                    <div class="w-3 h-0.5 bg-purple-500 mr-1"></div>
                    <span>Final</span>
                </div>
            </div>
        </div>
        
        <!-- Audio Player -->
        <div class="bg-white rounded-lg shadow-md p-4 mb-4">
            <h2 class="text-lg font-semibold mb-3">Audio Player</h2>
            <div class="flex items-center space-x-4 mb-4">
                <button id="playPauseBtn" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Play
                </button>
                <div class="flex-1">
                    <input type="range" id="seekBar" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" 
                           min="0" max="500" value="0">
                </div>
                <span id="timeDisplay" class="text-sm text-gray-600">0:00 / 0:00</span>
            </div>
            <audio id="audioPlayer" preload="auto">
                <source src="{audio_file}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
        
        <!-- Audio Waveform Chart -->
        <div class="bg-white rounded-lg shadow-md p-4 mb-4">
            <h2 class="text-lg font-semibold mb-3">Audio Waveform</h2>
            <div class="relative h-32">
                <canvas id="audioChart"></canvas>
            </div>
        </div>
        
        <!-- STT Provider Charts -->
        <div class="space-y-3">
            {generate_stt_chart_divs(chart_data.get("stt_data", {}))}
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data, default=str)};
        
        // Configure Chart.js defaults
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
        
        // Audio Chart
        const audioCtx = document.getElementById('audioChart').getContext('2d');
        const audioChart = new Chart(audioCtx, {{
            type: 'line',
            data: chartData.audio,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'linear',
                        min: 0,
                        max: chartData.duration
                    }},
                    y: {{
                        display: false,
                        min: -0.15,
                        max: 0.15
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        enabled: true,
                        mode: 'point',
                        intersect: false,
                        callbacks: {{
                            title: function(context) {{
                                if (context.length > 0) {{
                                    const point = context[0];
                                    return 'Time: ' + point.parsed.x.toFixed(2) + 's';
                                }}
                                return '';
                            }},
                            label: function(context) {{
                                const point = context.raw;
                                const text = point.text || '';
                                const type = context.dataset.label;
                                return [type + ':', '"' + text + '"'];
                            }}
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'point'
                }},
                animation: false
            }},
            plugins: [{{
                id: 'vadShading',
                beforeDraw: function(chart) {{
                    const ctx = chart.ctx;
                    const chartArea = chart.chartArea;
                    
                    // Draw VAD regions
                    chartData.vad_regions.forEach(region => {{
                        const startX = chart.scales.x.getPixelForValue(region.start);
                        const endX = chart.scales.x.getPixelForValue(region.end);
                        
                        ctx.save();
                        ctx.fillStyle = 'rgba(173, 216, 230, 0.3)';
                        ctx.fillRect(startX, chartArea.top, endX - startX, chartArea.bottom - chartArea.top);
                        ctx.restore();
                    }});
                    
                    // Draw speaking events with hover detection
                    chartData.speaking_events.forEach((event, index) => {{
                        const x = chart.scales.x.getPixelForValue(event.x);
                        const color = event.is_speaking ? 'green' : 'orange';
                        
                        ctx.save();
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 2;
                        ctx.setLineDash([]);
                        ctx.beginPath();
                        ctx.moveTo(x, chartArea.top);
                        ctx.lineTo(x, chartArea.bottom);
                        ctx.stroke();
                        ctx.restore();
                        
                        // Store event data for tooltip
                        if (!chart.vadEvents) chart.vadEvents = [];
                        chart.vadEvents[index] = {{
                            x: x,
                            time: event.x,
                            is_speaking: event.is_speaking,
                            left: x - 5,
                            right: x + 5,
                            top: chartArea.top,
                            bottom: chartArea.bottom
                        }};
                    }});
                    
                    // Draw playback position indicator
                    if (window.audioPlayer && !window.audioPlayer.paused && window.audioPlayer.currentTime) {{
                        const currentX = chart.scales.x.getPixelForValue(window.audioPlayer.currentTime);
                        
                        ctx.save();
                        ctx.strokeStyle = 'red';
                        ctx.lineWidth = 3;
                        ctx.setLineDash([]);
                        ctx.beginPath();
                        ctx.moveTo(currentX, chartArea.top);
                        ctx.lineTo(currentX, chartArea.bottom);
                        ctx.stroke();
                        ctx.restore();
                    }}
                }}
            }}]
        }});
        
        // Add custom tooltip for VAD events
        audioChart.canvas.addEventListener('mousemove', function(e) {{
            const rect = audioChart.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            if (audioChart.vadEvents) {{
                let hoveredEvent = null;
                audioChart.vadEvents.forEach(event => {{
                    if (x >= event.left && x <= event.right && y >= event.top && y <= event.bottom) {{
                        hoveredEvent = event;
                    }}
                }});
                
                if (hoveredEvent) {{
                    const tooltip = document.getElementById('vadTooltip') || createVadTooltip();
                    const eventType = hoveredEvent.is_speaking ? 'Speaking Start' : 'Speaking Stop';
                    tooltip.innerHTML = `Time: ${{hoveredEvent.time.toFixed(2)}}s<br>${{eventType}}`;
                    tooltip.style.left = (e.clientX + 10) + 'px';
                    tooltip.style.top = (e.clientY - 10) + 'px';
                    tooltip.style.display = 'block';
                }} else {{
                    const tooltip = document.getElementById('vadTooltip');
                    if (tooltip) tooltip.style.display = 'none';
                }}
            }}
        }});
        
        audioChart.canvas.addEventListener('mouseleave', function() {{
            const tooltip = document.getElementById('vadTooltip');
            if (tooltip) tooltip.style.display = 'none';
        }});
        
        function createVadTooltip() {{
            const tooltip = document.createElement('div');
            tooltip.id = 'vadTooltip';
            tooltip.style.position = 'absolute';
            tooltip.style.background = 'rgba(0,0,0,0.8)';
            tooltip.style.color = 'white';
            tooltip.style.padding = '5px 10px';
            tooltip.style.borderRadius = '4px';
            tooltip.style.fontSize = '12px';
            tooltip.style.pointerEvents = 'none';
            tooltip.style.zIndex = '1000';
            tooltip.style.display = 'none';
            document.body.appendChild(tooltip);
            return tooltip;
        }}
        
        // Audio Player Controls
        const audioPlayer = document.getElementById('audioPlayer');
        window.audioPlayer = audioPlayer; // Make globally accessible for playback indicator
        const playPauseBtn = document.getElementById('playPauseBtn');
        const seekBar = document.getElementById('seekBar');
        const timeDisplay = document.getElementById('timeDisplay');
        
        let currentTranscripts = {{}};
        let sttCharts = {{}};
        
        // Play/Pause functionality
        playPauseBtn.addEventListener('click', function() {{
            if (audioPlayer.paused) {{
                audioPlayer.play();
                playPauseBtn.textContent = 'Pause';
            }} else {{
                audioPlayer.pause();
                playPauseBtn.textContent = 'Play';
            }}
        }});
        
        // Seek bar functionality
        seekBar.addEventListener('input', function() {{
            const time = (seekBar.value / 500) * audioPlayer.duration;
            audioPlayer.currentTime = time;
            
            // Update transcript display and redraw chart to show position marker
            updateTranscriptDisplay(time);
            audioChart.update('none');
        }});
        
        // Update seek bar and time display
        audioPlayer.addEventListener('timeupdate', function() {{
            if (audioPlayer.duration) {{
                const progress = (audioPlayer.currentTime / audioPlayer.duration) * 500;
                seekBar.value = progress;
                
                const currentTime = formatTime(audioPlayer.currentTime);
                const totalTime = formatTime(audioPlayer.duration);
                timeDisplay.textContent = `${{currentTime}} / ${{totalTime}}`;
                
                // Update transcript display
                updateTranscriptDisplay(audioPlayer.currentTime);
                
                // Redraw audio chart to show playback position
                audioChart.update('none');
            }}
        }});
        
        // Reset play button when audio ends
        audioPlayer.addEventListener('ended', function() {{
            playPauseBtn.textContent = 'Play';
            clearAllTranscripts();
        }});
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        function updateTranscriptDisplay(currentTime) {{
            Object.keys(chartData.stt_data).forEach(provider => {{
                const events = chartData.stt_data[provider];
                let currentTranscript = null;
                
                // Find the most recent transcript at or before current time
                for (let i = events.length - 1; i >= 0; i--) {{
                    if (events[i].x <= currentTime) {{
                        currentTranscript = events[i];
                        break;
                    }}
                }}
                
                // Update transcript display for this provider
                if (currentTranscript && currentTranscript !== currentTranscripts[provider]) {{
                    currentTranscripts[provider] = currentTranscript;
                    showTranscriptOnChart(provider, currentTranscript);
                }} else if (!currentTranscript && currentTranscripts[provider]) {{
                    currentTranscripts[provider] = null;
                    clearTranscriptOnChart(provider);
                }}
            }});
        }}
        
        function showTranscriptOnChart(provider, transcript) {{
            const chart = sttCharts[provider];
            if (chart) {{
                // Clear previous transcript overlay
                clearTranscriptOnChart(provider);
                
                // Add transcript overlay
                const canvas = chart.canvas;
                const ctx = canvas.getContext('2d');
                const chartArea = chart.chartArea;
                
                // Store overlay for clearing later
                if (!chart.transcriptOverlay) {{
                    chart.transcriptOverlay = document.createElement('div');
                    chart.transcriptOverlay.style.position = 'absolute';
                    chart.transcriptOverlay.style.left = '50%';
                    chart.transcriptOverlay.style.top = '50%';
                    chart.transcriptOverlay.style.transform = 'translate(-50%, -50%)';
                    chart.transcriptOverlay.style.background = 'rgba(0,0,0,0.8)';
                    chart.transcriptOverlay.style.color = 'white';
                    chart.transcriptOverlay.style.padding = '8px 12px';
                    chart.transcriptOverlay.style.borderRadius = '4px';
                    chart.transcriptOverlay.style.fontSize = '14px';
                    chart.transcriptOverlay.style.maxWidth = '80%';
                    chart.transcriptOverlay.style.textAlign = 'center';
                    chart.transcriptOverlay.style.pointerEvents = 'none';
                    chart.transcriptOverlay.style.zIndex = '100';
                    canvas.parentElement.appendChild(chart.transcriptOverlay);
                }}
                
                const transcriptType = transcript.is_final ? 'Final' : 'Interim';
                const color = transcript.is_final ? '#8b5cf6' : '#ef4444';
                chart.transcriptOverlay.innerHTML = `
                    <div style="color: ${{color}}; font-weight: bold; margin-bottom: 4px;">${{transcriptType}}</div>
                    <div>"${{transcript.text}}"</div>
                `;
                chart.transcriptOverlay.style.display = 'block';
            }}
        }}
        
        function clearTranscriptOnChart(provider) {{
            const chart = sttCharts[provider];
            if (chart && chart.transcriptOverlay) {{
                chart.transcriptOverlay.style.display = 'none';
            }}
        }}
        
        function clearAllTranscripts() {{
            Object.keys(currentTranscripts).forEach(provider => {{
                currentTranscripts[provider] = null;
                clearTranscriptOnChart(provider);
            }});
        }}
        
        // STT Provider Charts
        {generate_stt_chart_js(chart_data.get("stt_data", {}), chart_data.get("duration", 10))}
    </script>
</body>
</html>'''

    # Write HTML file
    html_file = html_folder / output_filename
    with open(html_file, "w") as f:
        f.write(html_content)

    print(f"Interactive HTML chart created: {html_file}")


def generate_stt_chart_divs(stt_data: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate HTML divs for STT provider charts."""
    divs = []
    for provider in stt_data.keys():
        chart_id = f"stt{provider.lower()}Chart"
        divs.append(f'''
        <div class="bg-white rounded-lg shadow-md p-3">
            <h2 class="text-base font-semibold mb-2">{provider}</h2>
            <div class="relative h-20">
                <canvas id="{chart_id}"></canvas>
            </div>
        </div>''')
    return "".join(divs)


def generate_stt_chart_js(stt_data: Dict[str, List[Dict[str, Any]]], duration: float) -> str:
    """Generate JavaScript for STT provider charts."""
    js_code = []

    for provider, events in stt_data.items():
        chart_id = f"stt{provider.lower()}Chart"

        # Prepare datasets for interim and final transcripts
        interim_data = []
        final_data = []

        for event in events:
            point = {"x": event["x"], "y": 0, "text": event["text"], "timestamp": event["x"]}

            if event["is_final"]:
                final_data.append(point)
            else:
                interim_data.append(point)

        js_code.append(f"""
        // {provider} Chart
        const {chart_id.replace("Chart", "Ctx")} = document.getElementById('{chart_id}').getContext('2d');
        const {chart_id.replace("Chart", "")} = new Chart({chart_id.replace("Chart", "Ctx")}, {{
            type: 'scatter',
            data: {{
                datasets: [
                    {{
                        label: 'Interim Transcript',
                        data: {json.dumps(interim_data)},
                        backgroundColor: 'red',
                        borderColor: 'red',
                        pointRadius: 3,
                        showLine: false
                    }},
                    {{
                        label: 'Final Transcript',
                        data: {json.dumps(final_data)},
                        backgroundColor: 'purple',
                        borderColor: 'purple',
                        pointRadius: 5,
                        showLine: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'linear',
                        min: 0,
                        max: {duration}
                    }},
                    y: {{
                        display: false,
                        min: -0.5,
                        max: 0.5
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        enabled: true,
                        mode: 'point',
                        intersect: false,
                        callbacks: {{
                            title: function(context) {{
                                if (context.length > 0) {{
                                    const point = context[0];
                                    return 'Time: ' + point.parsed.x.toFixed(2) + 's';
                                }}
                                return '';
                            }},
                            label: function(context) {{
                                const point = context.raw;
                                const text = point.text || '';
                                const type = context.dataset.label;
                                return [type + ':', '"' + text + '"'];
                            }}
                        }}
                    }}
                }},
                interaction: {{
                    intersect: false,
                    mode: 'point'
                }},
                animation: false
            }},
            plugins: [{{
                id: 'vadShading{provider}',
                beforeDraw: function(chart) {{
                    const ctx = chart.ctx;
                    const chartArea = chart.chartArea;
                    
                    // Draw VAD regions
                    chartData.vad_regions.forEach(region => {{
                        const startX = chart.scales.x.getPixelForValue(region.start);
                        const endX = chart.scales.x.getPixelForValue(region.end);
                        
                        ctx.save();
                        ctx.fillStyle = 'rgba(173, 216, 230, 0.3)';
                        ctx.fillRect(startX, chartArea.top, endX - startX, chartArea.bottom - chartArea.top);
                        ctx.restore();
                    }});
                }}
            }}]
        }});
        
        // Store chart reference for transcript display
        sttCharts['{provider}'] = {chart_id.replace("Chart", "")};""")

    return "\n".join(js_code)


def main():
    parser = argparse.ArgumentParser(description="Create interactive HTML charts from JSONL files")
    parser.add_argument("folder", help="Folder containing audio.jsonl file (e.g., output/test)")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test-result.html",
        help="Output filename (default: test-result.html)",
    )

    args = parser.parse_args()

    # Set up paths
    folder_path = Path(args.folder)
    html_folder = folder_path
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
            provider_events = load_stt_timestamps(str(provider_path))

            if provider_events:
                stt_data[provider] = provider_events
                print(f"Loaded {len(provider_events)} {provider} transcript events")
            else:
                print(f"No {provider} transcript events found")

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

        # Create audio base64 data
        audio_file = create_audio_base64(audio_samples, args.sample_rate)

        # Prepare chart data
        chart_data = prepare_chart_data(
            timestamps, audio_samples, stt_data, speaking_events, args.sample_rate
        )

        # Create HTML page
        create_html_page(chart_data, args.folder, html_folder, audio_file, args.output)

        print("Interactive HTML visualization complete!")
        return 0

    except Exception as e:
        print(f"Error processing data: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
