import streamlit as st

class RealtimeEngine:
    def __init__(self):
        self.connected = True
        self.latency = 5  # ms
        
    def update(self):
        if self.connected:
            return True
        return False 