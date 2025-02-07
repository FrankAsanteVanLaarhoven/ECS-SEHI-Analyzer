import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class DataPersistence:
    def __init__(self, db_path: str = "data/sehi_analysis.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    analysis_type TEXT,
                    data BLOB,
                    metadata TEXT
                )
            """)
            
    def save_analysis(self, 
                     analysis_type: str, 
                     data: Dict[str, Any],
                     metadata: Optional[Dict] = None):
        """Save analysis results to database"""
        serialized_data = self._serialize_data(data)
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO analysis_results (analysis_type, data, metadata) VALUES (?, ?, ?)",
                (analysis_type, serialized_data, metadata_json)
            )
            
    def load_analysis(self, 
                     analysis_type: Optional[str] = None,
                     limit: int = 10) -> Dict[str, Any]:
        """Load analysis results from database"""
        query = "SELECT * FROM analysis_results"
        if analysis_type:
            query += f" WHERE analysis_type = '{analysis_type}'"
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query)
            results = cursor.fetchall()
            
        return [self._deserialize_result(result) for result in results]
    
    def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize data for storage"""
        return json.dumps({
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }).encode()
        
    def _deserialize_result(self, result) -> Dict[str, Any]:
        """Deserialize database result"""
        id_, timestamp, analysis_type, data, metadata = result
        return {
            'id': id_,
            'timestamp': datetime.fromisoformat(timestamp),
            'analysis_type': analysis_type,
            'data': json.loads(data.decode()),
            'metadata': json.loads(metadata)
        } 