import plotly.graph_objects as go
import numpy as np
import streamlit as st

class SpectralProfileVisualizer:
    def __init__(self):
        self.bands = {
            'visible': (400, 700),
            'near_ir': (700, 1400),
            'short_ir': (1400, 2500)
        }

    def visualize(self, wavelengths: np.ndarray, intensities: np.ndarray) -> None:
        """Visualize spectral profile."""
        try:
            # Ensure 1D arrays
            wavelengths = np.squeeze(wavelengths)
            intensities = np.squeeze(intensities)
            
            if wavelengths.ndim != 1 or intensities.ndim != 1:
                raise ValueError(f"Expected 1D arrays, got shapes: wavelengths {wavelengths.shape}, intensities {intensities.shape}")

            # Create spectral plot
            fig = go.Figure()
            
            # Add main spectrum
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=intensities,
                mode='lines',
                name='Spectrum',
                line=dict(color='cyan', width=2)
            ))

            # Add band regions
            for band_name, (start, end) in self.bands.items():
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor="rgba(128, 128, 128, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text=band_name.replace('_', ' ').title(),
                    annotation_position="top left"
                )

            # Update layout
            fig.update_layout(
                title="Spectral Profile",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (a.u.)",
                template="plotly_dark",
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add spectral analysis
            with st.expander("Spectral Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Peak Wavelength", f"{wavelengths[np.argmax(intensities)]:.0f} nm")
                    st.metric("Peak Intensity", f"{np.max(intensities):.2f}")
                with col2:
                    st.metric("Mean Intensity", f"{np.mean(intensities):.2f}")
                    st.metric("Spectral Range", f"{wavelengths[-1]-wavelengths[0]:.0f} nm")

        except Exception as e:
            st.error(f"Spectral profile visualization failed: {str(e)}")
            st.exception(e) 