import streamlit as st
from typing import Optional, List, Dict
import os

class ProtocolAssistant:
    def __init__(self):
        self.context = []
        self.available_providers = []
        self.env_loaded = False
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check available LLM providers and load environment if possible"""
        # Try to load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            self.env_loaded = True
        except ImportError:
            st.sidebar.warning("💡 Tip: Install python-dotenv for easier configuration")
        
        # Check OpenAI
        try:
            import openai
            self.openai = openai
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            if api_key:
                self.openai.api_key = api_key
                self.available_providers.append("OpenAI")
            else:
                st.sidebar.info("OpenAI API key not found")
        except ImportError:
            st.sidebar.info("OpenAI not installed")
            
        # Check Perplexity
        try:
            from perplexity import Client
            api_key = os.getenv("PERPLEXITY_API_KEY") or st.secrets.get("PERPLEXITY_API_KEY")
            if api_key:
                self.perplexity = Client(api_key=api_key)
                self.available_providers.append("Perplexity")
            else:
                st.sidebar.info("Perplexity API key not found")
        except ImportError:
            st.sidebar.info("Perplexity not installed")
            
    def render_assistant_interface(self):
        """Render AI assistant interface"""
        st.markdown("### 🤖 Protocol Assistant")
        
        # Show configuration help if no providers available
        if not self.available_providers:
            st.error("No AI providers configured")
            with st.expander("📝 Configuration Help"):
                st.markdown("""
                ### Setup Instructions
                
                1. Install required packages:
                ```bash
                pip install openai perplexity python-dotenv
                ```
                
                2. Configure API keys using one of these methods:
                
                **Option A: Using .env file** (Recommended)
                ```bash
                # Create .env file with:
                OPENAI_API_KEY=your_key_here
                PERPLEXITY_API_KEY=your_key_here
                ```
                
                **Option B: Using Streamlit Secrets**
                ```toml
                # In .streamlit/secrets.toml:
                OPENAI_API_KEY="your_key_here"
                PERPLEXITY_API_KEY="your_key_here"
                ```
                
                **Option C: Environment Variables**
                ```bash
                export OPENAI_API_KEY=your_key_here
                export PERPLEXITY_API_KEY=your_key_here
                ```
                """)
            return
            
        # Provider selection
        provider = st.selectbox(
            "Select AI Provider",
            self.available_providers,
            key="ai_provider"
        )
        
        # Protocol input
        protocol = st.text_area(
            "Enter Research Protocol",
            height=200,
            key="protocol_input",
            help="Enter your research protocol for AI analysis"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            analyze_button = st.button(
                "Analyze Protocol",
                key="analyze_protocol",
                type="primary"
            )
            
        with col2:
            st.markdown(f"Using: **{provider}**")
            
        if analyze_button and protocol:
            with st.spinner("Analyzing protocol..."):
                try:
                    if provider == "OpenAI":
                        analysis = self.analyze_with_openai(protocol)
                    else:
                        analysis = self.analyze_with_perplexity(protocol)
                        
                    if "error" not in analysis:
                        st.success("Analysis complete!")
                        
                        with st.expander("Analysis Results", expanded=True):
                            st.markdown("#### Key Points")
                            st.write(analysis["analysis"])
                            
                            if analysis.get("suggestions"):
                                st.markdown("#### Suggestions")
                                for suggestion in analysis["suggestions"]:
                                    st.info(suggestion)
                    else:
                        st.error(f"Analysis failed: {analysis['error']}")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        elif analyze_button:
            st.warning("Please enter a protocol to analyze")
            
    def analyze_with_openai(self, protocol_text: str) -> Dict:
        """Analyze protocol using OpenAI"""
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific protocol assistant."},
                    {"role": "user", "content": f"Analyze this protocol:\n{protocol_text}"}
                ]
            )
            return {
                "analysis": response.choices[0].message.content,
                "suggestions": self._extract_suggestions(response)
            }
        except Exception as e:
            return {"error": str(e)}
            
    def analyze_with_perplexity(self, protocol_text: str) -> Dict:
        """Analyze protocol using Perplexity"""
        try:
            response = self.perplexity.generate_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a scientific protocol assistant."},
                    {"role": "user", "content": f"Analyze this protocol:\n{protocol_text}"}
                ],
                model="pplx-7b-chat"
            )
            return {
                "analysis": response["choices"][0]["message"]["content"],
                "suggestions": self._extract_suggestions(response)
            }
        except Exception as e:
            return {"error": str(e)}
            
    def _extract_suggestions(self, response: Dict) -> List[str]:
        """Extract suggestions from LLM response"""
        try:
            if isinstance(response, dict) and "choices" in response:
                text = response["choices"][0]["message"]["content"]
            else:
                text = response.choices[0].message.content
            suggestions = []
            for line in text.split('\n'):
                if line.startswith('- ') or line.startswith('* '):
                    suggestions.append(line[2:])
            return suggestions
        except:
            return [] 