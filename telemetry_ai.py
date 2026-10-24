import json
import requests

class TelemetryFramework:
    def __init__(self, model_name="qwen2.5:7b"):
        self.url = "http://localhost:11434/api/chat"
        self.model_name = model_name
        self.system_prompt = (
            "You are an elite Sim Racing Coach analyzing digital rally telemetry data. "
            "Provide brief, actionable driving inputs or hardware changes based on the data. "
            "Use precise sim-racing terminology like FFB clipping and throttle linearity."
        )

    def analyze(self, telemetry_data):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze: {json.dumps(telemetry_data)}"}
            ],
            "options": {
                "num_thread": 4,    # Protects your CPU threads so your games don't stutter
                "temperature": 0.2
            },
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            return response.json()["message"]["content"]
        except Exception:
            return "Error: Could not talk to Ollama. Make sure the app or terminal command is running!"

if __name__ == "__main__":
    # Test data representing a rough lap on ice
    mock_lap = {
        "stage": "Sweden Ice",
        "ffb_clipping_detected": True,
        "throttle_input_linearity": 0.72,
        "time_loss_vs_ghost_seconds": +0.85
    }
    
    coach = TelemetryFramework()
    print("Sending lap metrics to local Qwen 2.5 model...")
    print("\n=== COACHING INSIGHTS ===")
    print(coach.analyze(mock_lap))