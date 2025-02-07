# ECS SEHI Analysis Dashboard

Advanced scientific analysis and visualization platform for ECS SEHI data.

## Features

- 📊 Real-time data visualization
- 🎮 Interactive sandbox environment
- 🤖 AI-powered analysis
- 👥 Team collaboration tools
- 🌐 Holographic visualization
- 🎥 Screen recording studio
- 🌱 Sustainability metrics
- 🔮 Quantum computing integration

## Deployment

This project is deployed using Render.com. The deployment configuration can be found in `render.yaml`.

### Requirements

- Python 3.9.18
- PortAudio (for audio processing)
- OpenAI API key (for AI features)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/FrankAsanteVanLaarhoven/ecs_sehi-analysis-dashboard.git
cd ecs_sehi-analysis-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the dashboard:
```bash
streamlit run src/dashboard/dashboard.py
```

### Deployment Configuration

The application is configured to deploy automatically on Render.com with the following specifications:

- Runtime: Python 3.9.18
- Region: Frankfurt
- Auto Deploy: Enabled
- Health Check: /_stcore/health

### Development

To contribute to this project:

1. Create a new branch
2. Make your changes
3. Submit a pull request

## Documentation

For detailed documentation, please visit the [Wiki](https://github.com/FrankAsanteVanLaarhoven/ecs_sehi-analysis-dashboard/wiki).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
