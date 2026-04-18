from agents import AdvisorAgent

# Initialize the system
agent_system = AdvisorAgent()

# Test Case 1: A flight likely to be delayed (e.g., Evening flight)
print("\n--- TEST CASE 1 ---")
result = agent_system.get_travel_advice(
    airline="DL",      # Delta
    origin="ATL",      # Atlanta
    dest="LAX",        # Los Angeles
    date="2024-12-24", # Christmas Eve (Busy!)
    time="18:30"       # 6:30 PM
)
print(result)

# Test Case 2: A flight likely to be on time (e.g., Early morning)
print("\n--- TEST CASE 2 ---")
result2 = agent_system.get_travel_advice(
    airline="AA",      # American
    origin="JFK",      # New York
    dest="MCO",        # Orlando
    date="2024-07-10",
    time="06:00"       # 6:00 AM
)
print(result2)