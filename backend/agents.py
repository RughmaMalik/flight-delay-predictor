import pandas as pd
import joblib
import os
import json
from datetime import datetime
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class AnalystAgent:
    """
    Agent 1: ML-based prediction and structured data retrieval
    Provides numerical predictions and historical statistics
    """
    def __init__(self):
        print("Initializing Analyst Agent...")
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, '../models')
        data_path = os.path.join(base_path, '../data/flights_dataset.csv')

        # Load ML artifacts
        try:
            self.model = joblib.load(os.path.join(model_path, 'flight_delay_model.pkl'))
            self.le_carrier = joblib.load(os.path.join(model_path, 'le_carrier.pkl'))
            self.le_origin = joblib.load(os.path.join(model_path, 'le_origin.pkl'))
            self.le_dest = joblib.load(os.path.join(model_path, 'le_dest.pkl'))
            self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        except FileNotFoundError as e:
            print(f"ERROR: Models not found - {e}")
            raise

        print("Loading historical flight data for retrieval...")

        # Load historical data
        df = pd.read_csv(data_path)

        # Distance mapping
        self.route_distances = (
            df.groupby(['origin', 'dest'])['distance']
            .mean()
            .to_dict()
        )

        # Historical delay reasons
        delay_cols = [
            "weather_delay", "carrier_delay", "late_aircraft_delay",
            "nas_delay", "security_delay"
        ]
        
        delayed_df = df[df["dep_delay"] > 15]

        # Route-specific delay patterns
        self.route_delay_reasons = (
            delayed_df.groupby(['origin', 'dest'])[delay_cols]
            .mean().to_dict(orient="index")
        )

        # Airline-specific delay patterns
        self.airline_delay_reasons = (
            delayed_df.groupby('op_unique_carrier')[delay_cols]
            .mean().to_dict(orient="index")
        )

        print("Analyst Agent Ready")

    def get_distance(self, origin, dest):
        """Retrieve route distance"""
        return self.route_distances.get((origin, dest), 1000.0)

    def predict_delay(self, airline, origin, dest, date_str, time_str):
        """Generate ML-based delay probability"""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            month = dt.month
            day = dt.day
            day_of_week = dt.weekday() + 1
            hour, minute = time_str.split(":")
            crs_dep_time = float(hour + minute)

            input_data = pd.DataFrame({
                'month': [month],
                'day_of_month': [day],
                'day_of_week': [day_of_week],
                'op_unique_carrier': [self.le_carrier.transform([airline])[0]],
                'origin': [self.le_origin.transform([origin])[0]],
                'dest': [self.le_dest.transform([dest])[0]],
                'crs_dep_time': [crs_dep_time],
                'distance': [self.get_distance(origin, dest)]
            })

            input_scaled = self.scaler.transform(input_data)
            prob = self.model.predict_proba(input_scaled)[0][1]
            return prob * 100
        except Exception as e:
            print(f"Prediction Error: {e}")
            return -1

    def get_historical_delay_reasons(self, airline, origin, dest, top_n=3):
        """Retrieve structured historical delay data"""
        reasons = {}
        
        # Route-specific data
        if (origin, dest) in self.route_delay_reasons:
            reasons.update(self.route_delay_reasons[(origin, dest)])
        
        # Airline-specific data
        if airline in self.airline_delay_reasons:
            for k, v in self.airline_delay_reasons[airline].items():
                reasons[k] = max(reasons.get(k, 0), v)

        if not reasons:
            return ["No historical data available"]

        # Sort and format
        sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        return [
            f"{r[0].replace('_', ' ').title()} ({r[1]:.1f} min avg)" 
            for r in sorted_reasons[:top_n]
        ]

# =========================================================================

# RAG RETRIEVER FUNCTION
def load_rag_retriever():
    """
    Loads the FAISS vector store for semantic search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        "flight_rag_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================================================

# ADVISORY AGENT WITH RAG
class AdvisorAgent:
    """
    Agent 2: RAG-powered advisory system
    Combines ML predictions with domain knowledge retrieval and LLM generation
    """
    def __init__(self, groq_api_key):
        self.analyst = AnalystAgent()
        self.rag_retriever = load_rag_retriever()
        
        if groq_api_key:
            self.client = Groq(api_key=groq_api_key)
        else:
            self.client = None
            print("No API key provided - LLM features disabled")

    def get_airlines(self):
        return list(self.analyst.le_carrier.classes_)

    def get_origin(self):
        return list(self.analyst.le_origin.classes_)

    def get_dest(self):
        return list(self.analyst.le_dest.classes_)

    def get_travel_advice(self, airline, origin, dest, date, time):
        """
        Main advisory function with RAG pipeline:
        1. RETRIEVE: Get ML prediction + historical stats + domain knowledge
        2. AUGMENT: Build context-rich prompt with all retrieved information
        3. GENERATE: LLM produces grounded, evidence-based advice
        """
        print(f"\n Analyzing flight {airline} {origin}→{dest}...")

        # ===== STEP 1: RETRIEVE STRUCTURED DATA =====
        risk_score = self.analyst.predict_delay(airline, origin, dest, date, time)
        historical_reasons = self.analyst.get_historical_delay_reasons(airline, origin, dest)

        if risk_score == -1:
            return {"error": "Invalid airline or airport code"}

        # Determine risk category
        if risk_score > 60:
            status = "HIGH RISK"
        elif risk_score > 30:
            status = "MODERATE RISK"
        else:
            status = "ON TIME EXPECTED"

        # ===== STEP 2: RETRIEVE DOMAIN KNOWLEDGE =====
        # Build search query from delay reasons
        primary_delay_type = historical_reasons[0].split('(')[0].strip().lower()
        
        rag_query = f"""
        flight delay {primary_delay_type} advice prevention mitigation 
        {origin} {dest} airline travel tips
        """
        
        print(f"Retrieving domain knowledge for: {primary_delay_type}")
        rag_docs = self.rag_retriever.invoke(rag_query)
        
        # Extract text from retrieved documents
        rag_context = "\n\n".join([doc.page_content for doc in rag_docs])

        # ===== STEP 3: AUGMENT PROMPT WITH ALL CONTEXT =====
        system_prompt = """
        You are an expert aviation travel advisor with access to:
        1. Real-time ML predictions
        2. Historical airline performance data
        3. Aviation industry best practices and domain knowledge

        Provide actionable, evidence-based travel advice grounded in the provided context.
        Do NOT invent facts. Only use information from the provided evidence.
        """

        user_prompt = f"""
        Analyze this flight and provide a risk assessment.

        === FLIGHT DETAILS ===
        Airline: {airline}
        Route: {origin} → {dest}
        Date: {date} at {time}

        === ML PREDICTION ===
        Delay Probability: {risk_score:.1f}%
        Risk Level: {status}

        === HISTORICAL DATA ===
        Top Delay Contributors: {", ".join(historical_reasons)}

        === DOMAIN KNOWLEDGE (Evidence Base) ===
        {rag_context}

        === TASK ===
        Based on the evidence provided above, generate a response in STRICT JSON format:

        {{
          "risk_interpretation": "2-3 sentence explanation of the risk level and what it means for the passenger",
          "key_factors": [
            "factor 1 from historical data",
            "factor 2 from historical data"
          ]
        }}

        Rules:
        - Ground interpretation in the provided evidence
        - Keep professional tone
        - NO markdown, NO backticks, NO extra commentary
        - Pure JSON only
        """

        # ===== STEP 4: GENERATE LLM RESPONSE =====
        llm_advice = {"error": "LLM unavailable"}
        
        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=600,
                )
                
                response_text = chat_completion.choices[0].message.content
                llm_advice = json.loads(response_text)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                llm_advice = {
                    "error": "Could not parse LLM response",
                    "raw_response": response_text
                }
            except Exception as e:
                print(f"LLM Error: {e}")
                llm_advice = {"error": str(e)}
        else:
            llm_advice = {"error": "No API key provided"}

        # ===== RETURN COMPLETE RESULT =====
        return {
            "prediction": status,
            "confidence": f"{risk_score:.1f}%",
            "historical_reasons": historical_reasons,
            "rag_advice": llm_advice,
            "evidence_sources": [doc.metadata.get("source", "unknown") for doc in rag_docs]
        }

    # ===== CHAT WITH USER =====
    def chat_with_user(self, user_message, chat_history, context_data):
        """
        RAG-powered Q&A agent for follow-up questions
        """
        if not self.client:
            return "Error: API key required for chat functionality"

        # Build semantic query from user's question
        rag_docs = self.rag_retriever.invoke(user_message)
        rag_context = "\n\n".join([doc.page_content for doc in rag_docs])

        # System instruction with flight context
        system_instruction = {
            "role": "system",
            "content": f"""
            You are a helpful aviation travel assistant.
            
            The user is reviewing this flight analysis:
            {json.dumps(context_data, indent=2)}
            
            Answer their question using:
            1. The flight analysis context above
            2. The domain knowledge provided below
            
            Domain Knowledge:
            {rag_context}
            
            Be helpful, specific, and ground answers in provided evidence.
            If you don't have enough information, say so.
            answer breifly and to the point.
            """
        }

        # Build message chain
        messages = [system_instruction] + chat_history + [
            {"role": "user", "content": user_message}
        ]

        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=400,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Chat Error: {str(e)}"

# =========================================================================

# QUESTION FILTERING FUNCTION
def is_question_relevant(question: str) -> bool:
    """
    Filter for relevant flight-related questions
    """
    question_lower = question.lower()
    
    relevant_keywords = [
        "flight", "delay", "airline", "airport", "weather", "risk",
        "alternative", "recommendation", "why", "what", "how",
        "carrier", "late", "cancel", "book", "time", "route",
        "compensation", "rebooking", "advice", "should", "better"
    ]
    
    return any(keyword in question_lower for keyword in relevant_keywords)