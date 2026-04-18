"""
Script to create a vector knowledge base for flight delay RAG system.
Run this ONCE before using the application.

Requirements:
pip install langchain-community faiss-cpu sentence-transformers
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Aviation domain knowledge for RAG
flight_knowledge = [
    # Weather-related delays
    """
    Weather Delays: Impact and Mitigation
    
    Weather is the leading cause of flight delays, accounting for 70% of all delays during peak seasons.
    Common weather factors include thunderstorms, fog, snow, and high winds.
    
    Passenger Advice:
    - Check weather forecasts for both departure and arrival cities 24-48 hours before travel
    - Book morning flights when weather is typically more stable
    - Avoid travel during hurricane season (June-November) in affected regions
    - Consider purchasing travel insurance for weather-related cancellations
    - Allow extra connection time during winter months
    
    Airlines cannot control weather delays, so compensation is typically not provided.
    However, they may offer rebooking options or accommodations depending on severity.
    """,
    
    # Carrier delays
    """
    Carrier Delays: Operational Issues
    
    Carrier delays are airline-specific operational problems including:
    - Maintenance issues and aircraft inspections
    - Crew scheduling problems and staffing shortages
    - Baggage loading delays
    - Aircraft cleaning and servicing
    - Fueling delays
    
    Passenger Rights:
    - For controllable delays over 3 hours, passengers may be entitled to compensation
    - Request meal vouchers for delays over 2 hours
    - Ask about rebooking on partner airlines
    - Document delay reasons for potential reimbursement claims
    
    Prevention Tips:
    - Choose airlines with better on-time performance records
    - Avoid the first flight on an aircraft's daily schedule (more prone to maintenance delays)
    - Join airline loyalty programs for priority rebooking
    """,
    
    # Late aircraft delays
    """
    Late Aircraft Delays: The Ripple Effect
    
    Late aircraft delays occur when the incoming aircraft is delayed, creating a cascading effect.
    These account for 25-30% of all delays and are more common during:
    - Peak travel times (holidays, summer)
    - Evening and late-night flights
    - Airports with tight turnaround schedules
    
    Risk Factors:
    - Short connection times between flights
    - Aircraft with multiple daily legs
    - Busy hub airports during rush hours
    
    Mitigation Strategies:
    - Book direct flights when possible
    - Choose flights earlier in the day (aircraft is less likely to be delayed from previous legs)
    - Avoid minimum connection times; add 30-60 minute buffer
    - Monitor your incoming aircraft's status using flight tracking apps
    - Consider non-hub routing for more reliable schedules
    """,
    
    # NAS delays
    """
    National Air System (NAS) Delays: Air Traffic Control Issues
    
    NAS delays are caused by:
    - Air traffic control constraints
    - Heavy traffic volume at airports
    - Airport construction or runway closures
    - Airspace congestion
    - Non-weather related issues like equipment outages
    
    High-Risk Scenarios:
    - Major hub airports (ATL, ORD, LAX, DFW, DEN)
    - Peak travel hours: 6-9 AM and 4-8 PM
    - Summer vacation season and holiday periods
    - Routes through busy airspace corridors
    
    Traveler Tips:
    - Book flights during off-peak hours (10 AM - 2 PM)
    - Avoid mega-hub connections when possible
    - Check FAA delay information before departure
    - Subscribe to airline alerts for real-time updates
    - Consider smaller regional airports with less congestion
    """,
    
    # Security delays
    """
    Security Delays: TSA and Safety Protocols
    
    Security delays are relatively rare but can occur due to:
    - Enhanced security screenings
    - Suspicious items or unattended baggage
    - TSA staffing shortages
    - Equipment malfunctions at checkpoints
    - Special security events or threats
    
    Prevention Measures:
    - Arrive at least 2 hours early for domestic flights (3 hours international)
    - Enroll in TSA PreCheck or CLEAR for expedited screening
    - Pack carry-ons properly: liquids in 3.4oz containers, laptops accessible
    - Avoid busy screening times: Monday mornings, Friday afternoons
    - Check TSA wait times online before heading to airport
    
    Note: Security delays are considered uncontrollable and typically not compensated.
    """,
    
    # Seasonal patterns
    """
    Seasonal Flight Delay Patterns
    
    Summer (June-August):
    - Highest delay rates due to thunderstorms and peak travel demand
    - Afternoon and evening flights most affected
    - Southern and Midwest routes particularly vulnerable to weather
    
    Winter (December-February):
    - Snow and ice delays in northern regions
    - Holiday travel creates capacity issues
    - De-icing procedures add 15-45 minutes to departures
    
    Spring (March-May):
    - Moderate delay rates
    - Tornado season affects central US routes
    - Generally the best time for reliable travel
    
    Fall (September-November):
    - Hurricane season through November on coastal routes
    - Thanksgiving period sees highest travel volume
    - Best travel period: September and early October
    """,
    
    # Route-specific insights
    """
    High-Risk Routes and Airports
    
    Airports with Highest Delay Rates:
    - Newark (EWR): Congestion and weather
    - Chicago O'Hare (ORD): Winter weather and volume
    - San Francisco (SFO): Fog and single-runway operations
    - LaGuardia (LGA): Space constraints and weather
    - Fort Lauderdale (FLL): Thunderstorms and traffic
    
    Most Reliable Airports:
    - Salt Lake City (SLC): Good weather and efficient operations
    - Seattle (SEA): Modern infrastructure
    - Portland (PDX): Consistent on-time performance
    - Phoenix (PHX): Favorable weather year-round
    
    High-Risk Connections:
    - Any connection through ORD, EWR, ATL during peak hours
    - Florida-bound flights in hurricane season
    - Midwest routes during winter months
    - Cross-country red-eye flights (aircraft positioning issues)
    """,
    
    # Time-of-day patterns
    """
    Optimal Flight Times for On-Time Performance
    
    Best Times to Fly:
    - Early morning flights (6-8 AM): 85% on-time rate
    - Mid-morning flights (9-11 AM): 78% on-time rate
    - Fewer delays as aircraft and crew are "fresh"
    
    Moderate Risk:
    - Midday flights (12-3 PM): 70% on-time rate
    - Early delays start accumulating
    
    Highest Risk:
    - Late afternoon (4-7 PM): 62% on-time rate
    - Evening flights (8-10 PM): 58% on-time rate
    - Cascading delays from earlier flights
    - Weather systems fully developed
    - Crew duty time limitations
    
    Strategy: Always book the earliest available flight for critical trips.
    """,
    
    # Compensation and rights
    """
    Passenger Rights and Compensation for Delays
    
    US Domestic Flights:
    - No federal requirement for compensation on domestic delays
    - Airlines must refund checked bag fees if bags are significantly delayed
    - Passengers can request meal vouchers for extended delays (typically 2+ hours)
    - Hotel accommodations may be provided for overnight delays (controllable delays only)
    
    International Flights (EU261 Regulation):
    - €250-600 compensation for delays over 3 hours (distance-dependent)
    - Applies to flights departing EU or arriving on EU carriers
    - Compensation not required for extraordinary circumstances (weather, ATC)
    
    What to Request During Delays:
    1. Rebooking on next available flight (same airline or partner)
    2. Meal vouchers for delays over 2 hours
    3. Hotel and ground transportation for overnight delays
    4. Written documentation of delay reason
    5. Contact information for customer relations
    
    Document Everything:
    - Take photos of delay screens
    - Keep all receipts
    - Note gate agent names and times
    - Request delay confirmation in writing
    """,
    
    # Alternative strategies
    """
    Alternative Travel Strategies When Delays Are Predicted
    
    High Delay Risk (60%+ probability):
    
    Option 1: Change Airlines
    - Check if other carriers have flights on same route
    - Consider connecting through different hubs
    - Price compare vs. rebooking fees
    
    Option 2: Adjust Travel Time
    - Move to earlier flight (if available)
    - Split journey over two days with overnight stop
    - Consider day-before travel for critical appointments
    
    Option 3: Alternative Transportation
    - For routes under 300 miles: consider train or car
    - Amtrak offers reliable service on certain corridors
    - Rental car one-way might be cost-effective
    
    Option 4: Different Airport
    - Check nearby airports (e.g., BWI instead of DCA)
    - Regional airports often have better on-time records
    - Factor in ground transportation time and cost
    
    Option 5: Travel Insurance
    - Purchase "cancel for any reason" coverage
    - Covers non-refundable bookings if you cancel
    - Must be purchased within 14 days of initial trip deposit
    """
]

def create_flight_rag_store():
    """
    Creates FAISS vector store from flight domain knowledge.
    """
    print("Creating Flight Delay RAG Knowledge Base...")
    
    # Create Document objects
    documents = [
        Document(page_content=text, metadata={"source": f"flight_knowledge_{i}"})
        for i, text in enumerate(flight_knowledge)
    ]
    
    print(f"Processing {len(documents)} knowledge documents...")
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save locally
    vectorstore.save_local("flight_rag_store")
    
    print("RAG store created successfully at './flight_rag_store'")
    print("You can now run the application with true RAG support!")

if __name__ == "__main__":
    create_flight_rag_store()