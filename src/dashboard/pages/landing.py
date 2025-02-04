import streamlit as st
import streamlit.components.v1 as components

def render_landing_page():
    """Render animated landing page with 3D globe transition."""
    st.markdown("""
    <style>
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Landing page styles */
        .landing-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, #1a1a1a 0%, #0a192f 100%);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            transition: all 0.5s ease-in-out;
        }
        
        .title-animation {
            font-size: 3.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 1s ease-out forwards;
            text-align: center;
            text-shadow: 0 0 10px rgba(255,255,255,0.3);
            line-height: 1.2;
        }
        
        .subtitle-animation {
            font-size: 1.3rem;
            margin-bottom: 2rem;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 1s ease-out 0.5s forwards;
            text-align: center;
            max-width: 800px;
            line-height: 1.5;
        }
        
        .copyright {
            position: fixed;
            bottom: 20px;
            width: 100%;
            text-align: center;
            font-size: 1rem;
            opacity: 0;
            animation: fadeIn 1s ease-out 1.5s forwards;
            color: rgba(255,255,255,0.8);
            padding: 10px;
            background: rgba(0,0,0,0.3);
            backdrop-filter: blur(5px);
        }
        
        .author-info {
            font-weight: 500;
            color: white;
            margin-top: 5px;
        }
        
        .copyright-symbol {
            font-size: 1.2rem;
            margin: 0 5px;
        }
        
        /* Enhanced button styling */
        .enter-button {
            width: 300px;
            height: 60px;
            background: linear-gradient(45deg, #1e3a8a, #2563eb);
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 1.2rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 1s ease-out 1s forwards;
        }
        
        .enter-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            background: linear-gradient(45deg, #2563eb, #3b82f6);
        }
        
        .enter-button:before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.6s ease;
            z-index: -1;
        }
        
        .enter-button:hover:before {
            width: 300%;
            height: 300%;
        }
        
        .enter-button.clicked {
            transform: scale(0.95);
            pointer-events: none;
            background: linear-gradient(45deg, #2563eb, #3b82f6);
        }
        
        .landing-container.transitioning {
            transform: scale(1.1);
            opacity: 0;
            pointer-events: none;
        }
        
        .globe-container {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) scale(0.8);
            width: 100%;
            height: 100%;
            opacity: 0;
            transition: all 1s ease-in-out;
            pointer-events: none;
        }
        
        .globe-container.visible {
            opacity: 1;
            transform: translate(-50%, -50%) scale(1);
        }

        /* Particle animation */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
    </style>
    
    <div class="particles" id="particles-js"></div>
    """, unsafe_allow_html=True)

    # Add particle.js
    components.html("""
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
        <script>
            particlesJS('particles-js',
              {
                "particles": {
                  "number": {
                    "value": 80,
                    "density": {
                      "enable": true,
                      "value_area": 800
                    }
                  },
                  "color": {
                    "value": "#ffffff"
                  },
                  "shape": {
                    "type": "circle"
                  },
                  "opacity": {
                    "value": 0.5,
                    "random": false
                  },
                  "size": {
                    "value": 3,
                    "random": true
                  },
                  "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#ffffff",
                    "opacity": 0.4,
                    "width": 1
                  },
                  "move": {
                    "enable": true,
                    "speed": 2,
                    "direction": "none",
                    "random": false,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false
                  }
                },
                "interactivity": {
                  "detect_on": "canvas",
                  "events": {
                    "onhover": {
                      "enable": true,
                      "mode": "repulse"
                    },
                    "onclick": {
                      "enable": true,
                      "mode": "push"
                    },
                    "resize": true
                  }
                },
                "retina_detect": true
              }
            );
        </script>
    """, height=0)

    # Initialize session state for transitions
    if 'transition_state' not in st.session_state:
        st.session_state.transition_state = 'landing'

    with st.container():
        st.markdown("""
        <div class="landing-container" id="landingContainer">
            <h1 class="title-animation">ECS & SEHI Analysis Platform</h1>
            <p class="subtitle-animation">
                Engineered Carbon Surfaces & Secondary Electron Hyperspectral Imaging<br>
                <span style="font-size: 0.9em; opacity: 0.8;">Advanced Surface Analysis Technology</span>
            </p>
            <button class="enter-button" id="enterButton" onclick="handleEnter()">Enter Platform</button>
            <div class="copyright">
                Inspired by Johnson Matthey & University of Sheffield<br>
                <div class="author-info">
                    Developed by Frank Van Laarhoven<br>
                    <span class="copyright-symbol">©</span> 2025 All Rights Reserved
                </div>
            </div>
        </div>

        <script>
        function handleEnter() {
            const button = document.getElementById('enterButton');
            const container = document.getElementById('landingContainer');
            
            // Visual feedback
            button.classList.add('clicked');
            
            // Fade out landing page
            setTimeout(() => {
                container.style.opacity = '0';
                container.style.transform = 'scale(1.1)';
            }, 300);
            
            // Show globe transition
            setTimeout(() => {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: {'show_globe': true}
                }, '*');
            }, 800);
            
            // Navigate to dashboard
            setTimeout(() => {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: {'show_dashboard': true}
                }, '*');
            }, 3800);
        }
        </script>
        """, unsafe_allow_html=True)

        # Handle globe transition
        if st.session_state.get('show_globe', False):
            st.markdown('<div class="globe-container visible">', unsafe_allow_html=True)
            components.html("""
                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script>
                    // Set up Three.js scene
                    const scene = new THREE.Scene();
                    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                    const renderer = new THREE.WebGLRenderer({ antialias: true });
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    document.body.appendChild(renderer.domElement);

                    // Add a globe
                    const geometry = new THREE.SphereGeometry(5, 32, 32);
                    const texture = new THREE.TextureLoader().load('https://threejs.org/examples/textures/earth_atmos_2048.jpg');
                    const material = new THREE.MeshBasicMaterial({ map: texture });
                    const globe = new THREE.Mesh(geometry, material);
                    scene.add(globe);

                    // Position the camera
                    camera.position.z = 10;

                    // Animation loop
                    function animate() {
                        requestAnimationFrame(animate);
                        globe.rotation.y += 0.005;
                        renderer.render(scene, camera);
                    }
                    animate();
                </script>
            """, height=800)
            st.markdown('</div>', unsafe_allow_html=True)

def handle_landing_messages():
    """Handle messages from the landing page JavaScript."""
    components.html("""
        <script>
        window.addEventListener('message', function(e) {
            const data = e.data;
            
            if (data.type === 'streamlit:setComponentValue') {
                const value = data.value;
                
                if (value.show_globe) {
                    // Update session state via Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {'show_globe': true}
                    }, '*');
                }
                
                if (value.show_dashboard) {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {'show_dashboard': true}
                    }, '*');
                    
                    // Reload page to show dashboard
                    setTimeout(() => {
                        window.location.reload();
                    }, 500);
                }
            }
        });
        </script>
    """, height=0)

def main():
    """Main function to render the landing page and handle transitions."""
    render_landing_page()
    handle_landing_messages()

    if st.session_state.get('show_dashboard', False):
        st.write("Welcome to the Dashboard!")
        # Add your dashboard and navigation page code here

if __name__ == "__main__":
    main()