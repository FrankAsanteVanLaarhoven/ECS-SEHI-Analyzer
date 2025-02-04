import subprocess
import webbrowser
import time

def main():
    """Launch the Streamlit app on port 8502."""
    print("Starting SEHI Analysis Dashboard...")
    
    # Start the Streamlit server
    process = subprocess.Popen([
        "streamlit", "run",
        "streamlit_app.py",
        "--server.port", "8502",
        "--server.address", "localhost"
    ])
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Open the browser
    webbrowser.open("http://localhost:8502")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\nShutting down server...")

if __name__ == "__main__":
    main() 