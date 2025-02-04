import streamlit as st
import numpy as np
from scipy import signal
import threading
import time
import wave
import struct
import tempfile
import os
import pandas as pd
import streamlit.components.v1 as components

class SoundTherapy:
    """Sound therapy and binaural beats generator."""
    
    def __init__(self):
        self.sample_rate = 44100
        self.is_playing = False
        self.current_thread = None
        
        # Add nature sounds
        self.nature_sounds = {
            "None": None,
            "Ocean Waves": "ocean",
            "Rain Forest": "rainforest",
            "White Noise": "whitenoise",
            "Pink Noise": "pinknoise",
            "Brown Noise": "brownnoise"
        }
        
        # Enhanced frequency presets
        self.presets = {
            "Focus & Clarity": {
                "carrier": 432,  # Hz
                "beta": 15,      # Hz
                "description": "Enhances concentration and mental clarity",
                "recommended_background": "White Noise"
            },
            "Happiness & Joy": {
                "carrier": 528,  # Hz (Solfeggio frequency)
                "theta": 7.83,   # Hz (Schumann resonance)
                "description": "Promotes feelings of joy and well-being",
                "recommended_background": "Ocean Waves"
            },
            "Deep Flow": {
                "carrier": 396,  # Hz
                "alpha": 10,     # Hz
                "description": "Optimal state for flow and creativity",
                "recommended_background": "Pink Noise"
            },
            "Serotonin Boost": {
                "carrier": 440,  # Hz
                "theta": 6,      # Hz
                "description": "May help with serotonin production",
                "recommended_background": "Rain Forest"
            },
            "Dopamine Release": {
                "carrier": 512,  # Hz
                "alpha": 8,      # Hz
                "description": "Associated with reward and pleasure",
                "recommended_background": "Ocean Waves"
            },
            "Endorphin Flow": {
                "carrier": 639,  # Hz
                "theta": 5,      # Hz
                "description": "May promote natural endorphin release",
                "recommended_background": "Rain Forest"
            },
            # New presets
            "Deep Meditation": {
                "carrier": 174,  # Hz (Solfeggio frequency)
                "delta": 2,      # Hz
                "description": "Facilitates deep meditative states",
                "recommended_background": "Brown Noise"
            },
            "Healing & Recovery": {
                "carrier": 417,  # Hz (Solfeggio frequency)
                "theta": 6,      # Hz
                "description": "Supports cellular regeneration and healing",
                "recommended_background": "Pink Noise"
            },
            "Stress Relief": {
                "carrier": 396,  # Hz
                "alpha": 8,      # Hz
                "description": "Reduces anxiety and promotes calmness",
                "recommended_background": "Ocean Waves"
            },
            "Creative Flow": {
                "carrier": 852,  # Hz (Solfeggio frequency)
                "alpha": 10,     # Hz
                "description": "Enhances creative thinking and intuition",
                "recommended_background": "Rain Forest"
            }
        }

    def generate_binaural_beat(self, base_freq, beat_freq, duration):
        """Generate binaural beat frequencies."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        left_channel = np.sin(2 * np.pi * base_freq * t)
        right_channel = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
        return np.vstack((left_channel, right_channel)).T

    def generate_nature_sound(self, sound_type, duration):
        """Generate nature sound backgrounds."""
        if sound_type == "whitenoise":
            return np.random.normal(0, 0.1, int(self.sample_rate * duration))
        elif sound_type == "pinknoise":
            # Generate pink noise using 1/f spectrum
            f = np.fft.fftfreq(int(self.sample_rate * duration))
            f = np.abs(f)
            f[0] = 1e-6  # Avoid division by zero
            psd = 1.0 / f
            noise = np.random.normal(0, 1, int(self.sample_rate * duration))
            noise_fft = np.fft.fft(noise)
            noise_fft *= np.sqrt(psd)
            return np.real(np.fft.ifft(noise_fft)) * 0.1
        elif sound_type == "brownnoise":
            # Generate brown noise using integrated white noise
            white = np.random.normal(0, 1, int(self.sample_rate * duration))
            return np.cumsum(white) * 0.1 / np.sqrt(self.sample_rate)
        return None

    def create_wave_file(self, frequency_data, volume=0.5, duration=300, background_sound=None):
        """Create enhanced WAV file with binaural beats and nature sounds."""
        try:
            # Generate binaural beats
            sound_data = self.generate_binaural_beat(
                frequency_data["carrier"],
                frequency_data.get("beta", 
                    frequency_data.get("alpha", 
                    frequency_data.get("theta",
                    frequency_data.get("delta", 7)))),
                duration
            )
            
            # Add background nature sound if selected
            if background_sound:
                nature = self.generate_nature_sound(background_sound, duration)
                if nature is not None:
                    # Add nature sound to both channels
                    nature_stereo = np.vstack((nature, nature)).T
                    sound_data = sound_data * 0.7 + nature_stereo * 0.3
            
            # Scale the volume
            sound_data = sound_data * volume
            
            # Apply fade in/out
            fade_duration = min(3.0, duration * 0.1)  # 3 seconds or 10% of duration
            fade_length = int(fade_duration * self.sample_rate)
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            sound_data[:fade_length] *= fade_in[:, np.newaxis]
            sound_data[-fade_length:] *= fade_out[:, np.newaxis]
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                
                sound_data_int = (sound_data * 32767).astype(np.int16)
                wav_file.writeframes(sound_data_int.tobytes())
            
            return temp_file.name
            
        except Exception as e:
            st.error(f"Error creating audio file: {str(e)}")
            return None

    def create_persistent_player(self):
        """Create a persistent audio player that stays active across tabs."""
        if 'audio_session' not in st.session_state:
            st.session_state.audio_session = {
                'is_playing': False,
                'current_preset': None,
                'volume': 0.5,
                'background': None,
                'audio_bytes': None,
                'duration': 0
            }
        
        return st.session_state.audio_session

def create_persistent_stream():
    """Create a persistent audio stream container."""
    # Initialize player state if not exists
    if 'player_minimized' not in st.session_state:
        st.session_state.player_minimized = False
        
    if 'stream_container' not in st.session_state:
        st.session_state.stream_container = {
            'active_stream': None,
            'stream_type': None,
            'stream_url': None,
            'is_playing': False,
            'volume': 0.5,
            'position': 0
        }
    return st.session_state.stream_container

def handle_youtube_url(url):
    """Process YouTube URL for audio-only streaming."""
    video_id = url.split('v=')[-1].split('&')[0]
    # Use YouTube's audio-only embed with a minimal player
    return f"https://www.youtube-nocookie.com/embed/{video_id}?autoplay=1&controls=1&showinfo=0&modestbranding=1&rel=0&enablejsapi=1&widgetid=1&vq=tiny"

def handle_spotify_url(url):
    """Process Spotify URL for embedding."""
    # Extract track/playlist ID from URL
    if 'track' in url:
        return f"track/{url.split('track/')[-1]}"
    elif 'playlist' in url:
        return f"playlist/{url.split('playlist/')[-1]}"
    return url

def create_audio_player(stream_url, stream_type, mode='audio'):
    """Create a player with toggle between audio-only and normal streaming."""
    if stream_type == 'youtube':
        if mode == 'audio':
            return f"""
            <div class="audio-only-player">
                <audio id="audio-player" autoplay>
                    <source src="{stream_url}" type="audio/mp3">
                </audio>
                <script>
                    const player = document.getElementById('audio-player');
                    player.style.height = '40px';
                    player.style.width = '100%';
                </script>
            </div>
            """
        else:
            return f"""
            <div class="video-player">
                <iframe width="100%" 
                        height="200" 
                        src="{stream_url}"
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                </iframe>
            </div>
            """
    elif stream_type == 'spotify':
        if mode == 'audio':
            return f"""
            <iframe src="https://open.spotify.com/embed/{stream_url}" 
                    width="100%" 
                    height="80" 
                    frameborder="0" 
                    allowtransparency="true" 
                    allow="encrypted-media"
                    style="background: transparent;">
            </iframe>
            """
        else:
            return f"""
            <iframe src="https://open.spotify.com/embed/{stream_url}" 
                    width="100%" 
                    height="352" 
                    frameborder="0" 
                    allowtransparency="true" 
                    allow="encrypted-media"
                    style="background: transparent;">
            </iframe>
            """

def render_floating_player():
    """Render floating player with mode toggle."""
    stream_container = create_persistent_stream()
    
    # Initialize player mode if not exists
    if 'player_mode' not in st.session_state:
        st.session_state.player_mode = 'audio'
    
    # Check if we should show the player
    if ('audio_session' in st.session_state and 
        (st.session_state.audio_session['audio_bytes'] or stream_container['stream_url'])):
        
        floating_container = st.container()
        
        with floating_container:
            st.markdown(
                """
                <style>
                    .floating-player {
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        background-color: rgba(38, 39, 48, 0.9);
                        padding: 8px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                        z-index: 999;
                        transition: all 0.3s ease;
                        backdrop-filter: blur(10px);
                    }
                    .floating-player.minimized {
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        overflow: hidden;
                    }
                    .floating-player.expanded {
                        width: 250px;
                    }
                    .audio-only-player {
                        background: transparent;
                        height: 40px;
                        overflow: hidden;
                    }
                    .player-controls {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        margin-top: 4px;
                    }
                    .now-playing {
                        font-size: 12px;
                        opacity: 0.8;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Now safe to use player_minimized as it's initialized
            player_class = "floating-player minimized" if st.session_state.player_minimized else "floating-player expanded"
            st.markdown(f'<div class="{player_class}">', unsafe_allow_html=True)
            
            # Add position control
            if 'player_position' not in st.session_state:
                st.session_state.player_position = {'x': 20, 'y': 20}
            
            # Add position to style
            st.markdown(
                f"""
                <style>
                .floating-player {{
                    right: {st.session_state.player_position['x']}px;
                    bottom: {st.session_state.player_position['y']}px;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Minimize/Maximize button
            if st.button("🔄", key="toggle_player"):
                st.session_state.player_minimized = not st.session_state.player_minimized
                st.rerun()
            
            if not st.session_state.player_minimized:
                # Add mode toggle
                cols = st.columns([3, 1])
                with cols[1]:
                    if st.button("🎧" if st.session_state.player_mode == 'normal' else "🎬"):
                        st.session_state.player_mode = 'audio' if st.session_state.player_mode == 'normal' else 'normal'
                        st.rerun()
                
                if stream_container['stream_url']:
                    # Player with current mode
                    components.html(
                        create_audio_player(
                            stream_container['stream_url'],
                            stream_container['stream_type'],
                            st.session_state.player_mode
                        ),
                        height=250 if st.session_state.player_mode == 'normal' else 80
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_streaming_services():
    """Render streaming services with mode toggle."""
    stream_container = create_persistent_stream()
    
    st.markdown("### Streaming Services")
    
    # Add mode toggle in the main interface
    cols = st.columns([3, 1])
    with cols[1]:
        st.markdown("### Mode")
        mode = st.radio(
            "",
            ["Audio Only", "Full Player"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.player_mode = 'audio' if mode == "Audio Only" else 'normal'
    
    # Service selection
    service = st.selectbox(
        "Select Service",
        ["YouTube Music", "Spotify", "Custom Stream"]
    )
    
    if service == "YouTube Music":
        url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=..."
        )
        if url and st.button("Play"):
            stream_container.update({
                'stream_url': handle_youtube_url(url),
                'stream_type': 'youtube',
                'is_playing': True,
                'position': 0,
                'current_track': url.split('/')[-1]
            })
            
            # Preview current selection
            st.markdown("### Preview")
            components.html(
                create_audio_player(
                    stream_container['stream_url'],
                    'youtube',
                    st.session_state.player_mode
                ),
                height=250 if st.session_state.player_mode == 'normal' else 80
            )

def render_sound_therapy():
    """Render the sound therapy interface."""
    therapy = SoundTherapy()
    audio_session = therapy.create_persistent_player()
    
    # Main interface
    st.markdown("# Neural Frequency Therapy")
    
    # Create tabs for different audio sources
    source_tabs = st.tabs([
        "Frequency Therapy",
        "Streaming Services",
        "Playlist Manager"
    ])
    
    with source_tabs[0]:
        # Original frequency therapy interface
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### Frequency Presets")
            selected_preset = st.selectbox(
                "Choose your therapy",
                list(therapy.presets.keys())
            )
            
            # Show preset description
            st.info(therapy.presets[selected_preset]["description"])
            
            # Volume control
            volume = st.slider(
                "Volume",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            # Duration selection
            duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                max_value=60,
                value=5
            )
            
            # Add background sound selection
            background_sound = st.selectbox(
                "Background Sound",
                list(therapy.nature_sounds.keys()),
                index=0,
                help="Add natural background sounds to enhance the experience"
            )
            
            if background_sound != "None":
                background_volume = st.slider(
                    "Background Volume",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )
            
            # Modified Generate button
            if st.button("Generate & Play", type="primary"):
                with st.spinner("Generating therapeutic audio..."):
                    wav_file = therapy.create_wave_file(
                        therapy.presets[selected_preset],
                        volume,
                        duration * 60,
                        therapy.nature_sounds[background_sound]
                    )
                    
                    if wav_file:
                        try:
                            with open(wav_file, 'rb') as f:
                                audio_bytes = f.read()
                                
                                # Update session state
                                st.session_state.audio_session.update({
                                    'is_playing': True,
                                    'current_preset': selected_preset,
                                    'volume': volume,
                                    'background': background_sound,
                                    'audio_bytes': audio_bytes,
                                    'duration': duration
                                })
                                
                                # Download button
                                st.download_button(
                                    "Download Audio",
                                    audio_bytes,
                                    file_name=f"{selected_preset.lower().replace(' ', '_')}_therapy.wav",
                                    mime="audio/wav"
                                )
                        except Exception as e:
                            st.error(f"Error playing audio: {str(e)}")
                        finally:
                            try:
                                os.unlink(wav_file)
                            except:
                                pass
        
        with col2:
            st.markdown("### Frequency Information")
            
            # Create tabs for different information views
            info_tabs = st.tabs(["Wave Pattern", "Benefits", "Usage Guide"])
            
            with info_tabs[0]:
                # Show frequency wave visualization
                t = np.linspace(0, 0.5, 1000)
                freq_data = therapy.presets[selected_preset]
                carrier = np.sin(2 * np.pi * freq_data["carrier"] * t)
                
                st.line_chart(pd.DataFrame({
                    'Carrier Wave': carrier,
                    'Modulation': np.sin(2 * np.pi * 
                        freq_data.get("beta", 
                        freq_data.get("alpha", 
                        freq_data.get("theta", 7))) * t)
                }))
            
            with info_tabs[1]:
                st.markdown("""
                ### Potential Benefits
                - Enhanced focus and concentration
                - Improved mood and emotional well-being
                - Better stress management
                - Increased creativity
                - Deeper relaxation
                
                ### Frequency Effects
                - **Beta (12-30 Hz)**: Focus, cognition, alertness
                - **Alpha (8-12 Hz)**: Relaxation, creativity, flow
                - **Theta (4-8 Hz)**: Deep relaxation, meditation
                """)
            
            with info_tabs[2]:
                st.markdown("""
                ### How to Use
                1. Choose a frequency preset based on your desired state
                2. Use headphones for optimal binaural beat effect
                3. Find a comfortable, quiet space
                4. Start with short sessions (5-10 minutes)
                5. Adjust volume to a comfortable level
                
                ### Best Practices
                - Use stereo headphones
                - Keep volume moderate
                - Stay hydrated
                - Take breaks between sessions
                """)
    
    with source_tabs[1]:
        render_streaming_services()
    
    with source_tabs[2]:
        st.markdown("### Playlist Manager")
        
        # Create playlist
        st.text_input("Playlist Name", placeholder="My Wellness Mix")
        
        # Add tracks from different sources
        st.multiselect(
            "Add Tracks",
            [
                "Focus & Clarity (432 Hz)",
                "Deep Flow (396 Hz)",
                "Spotify: Deep Focus",
                "YouTube: Study Music",
                "Apple Music: Concentration Mix"
            ]
        )
        
        # Save playlist
        if st.button("Save Playlist"):
            st.success("Playlist saved!") 