import streamlit as st

class DataSyncManager:
    def __init__(self):
        self.sync_status = "Connected"
        self.last_sync = None
        
    def sync_data(self, data):
        self.last_sync = data
        return True 