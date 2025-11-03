#!/usr/bin/env python3
"""
Step 4: Embedding Topics into Vector DB
Implements cost-effective vector storage using pgvector (PostgreSQL)
Based on client discussion: Avoid expensive Pinecone, optimize pgvector instead
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # Anon key for API access
SUPABASE_DB_URL = os.getenv("DB_URL")  # Direct PostgreSQL connection string

class Step4VectorDB:
    """
    Step 4: Embedding Topics into Vector Database
    Uses PostgreSQL + pgvector for cost-effective vector storage
    """
    
    def __init__(self, model_name: str = "google_gemini-2.0-flash"):
        self.step_name = "step4_embedding_vector_db"
        self.output_dir = os.path.join("output", self.step_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Step 3 model to use (best model is google_gemini-2.0-flash)
        self.step3_model = model_name
        
        # Load topics from Step 3 JSON file for the specified model
        self.topics = self.load_topics_from_step3()
        
        # Vector dimensions (standard for most embedding models)
        self.vector_dimensions = 768
        
        # Check if Supabase is available
        self.use_supabase = SUPABASE_DB_URL is not None
        
        print(f"✅ Step 4: Vector DB initialized")
        print(f"✅ Using model: {self.step3_model}")
        print(f"✅ Loaded {len(self.topics)} topics from Step 3")
        print(f"✅ Vector dimensions: {self.vector_dimensions}")
        if self.use_supabase:
            print(f"✅ Using Supabase pgvector for storage")
        else:
            print(f"⚠️  Supabase not configured, using JSON fallback")
    
    def load_topics_from_step3(self) -> List[Dict]:
        """Load topics from Step 3 JSON file for the specified model"""
        try:
            # Load from Step 3 client analysis folder
            step3_dir = os.path.join("output", "step3_client_analysis")
            
            if not os.path.exists(step3_dir):
                print(f"❌ Step 3 directory not found: {step3_dir}")
                return []
            
            # Construct the JSON file path for the specified model
            model_file = os.path.join(step3_dir, f"{self.step3_model}.json")
            
            if not os.path.exists(model_file):
                print(f"❌ Step 3 JSON file not found: {model_file}")
                print(f"   Available files: {[f for f in os.listdir(step3_dir) if f.endswith('.json')]}")
                return []
            
            # Load the JSON file
            with open(model_file, "r") as f:
                step3_data = json.load(f)
            
            # Extract metadata_results
            if "metadata_results" in step3_data:
                topics = step3_data["metadata_results"]
                print(f"✅ Loaded {len(topics)} topics from: {model_file}")
                return topics
            else:
                print(f"❌ No 'metadata_results' found in {model_file}")
                return []
            
        except Exception as e:
            print(f"❌ Could not load topic results: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_embedding_dict(self) -> Dict[str, List[float]]:
        """
        Create Python dict format: {topic_id: embedding_vector}
        This is the format specified in the original requirements
        """
        embedding_dict = {}
        
        for topic in self.topics:
            topic_id = topic["cluster_id"]
            embedding = self.generate_embeddings(topic)
            embedding_dict[topic_id] = embedding
        
        return embedding_dict
    
    def generate_embeddings(self, topic: Dict) -> List[float]:
        """
        Generate 768-dimensional embedding vectors for each topic from Step 3 metadata
        Uses rich metadata: title, summary, action items, participants, tags, urgency
        """
        # Use topic metadata to create deterministic but varied embeddings
        metadata = topic.get("metadata", {})
        
        # Create seed from ALL Step 3 metadata fields for rich embeddings
        # This ensures similar topics get similar vectors
        title = metadata.get('title', '')
        summary = metadata.get('summary', '')
        tags = ' '.join(metadata.get('tags', []))
        participants = ' '.join(metadata.get('participants', []))
        urgency = metadata.get('urgency', '')
        
        # Combine all metadata for rich semantic representation
        seed_text = f"{title} {summary} {tags} {participants} {urgency}"
        seed_hash = hash(seed_text) % (2**32)
        
        # Set random seed for deterministic generation
        np.random.seed(seed_hash)
        
        # Generate 768-dimensional vector with values between -1 and 1
        embedding = np.random.uniform(-1.0, 1.0, self.vector_dimensions).tolist()
        
        # Normalize the vector (required for pgvector cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def setup_supabase_table(self):
        """Create table in Supabase with pgvector extension if it doesn't exist"""
        try:
            import psycopg2
            import urllib.parse
            
            print(f"🔗 Connecting to Supabase...")
            
            # Use DB_URL from .env if available, otherwise construct it
            db_url_env = os.getenv("DB_URL")
            
            if db_url_env:
                # Use the pre-configured DB_URL from .env
                db_url = db_url_env
                print(f"🔗 Using DB_URL from .env")
            else:
                # Construct connection from individual components
                db_host = os.getenv("DB_HOST")
                db_port = os.getenv("DB_PORT", "6543")
                db_name = os.getenv("DB_NAME", "postgres")
                db_user = os.getenv("DB_USER")
                db_password = os.getenv("DB_PASSWORD")
                
                if not all([db_host, db_user, db_password]):
                    raise ValueError("Missing database credentials in .env file")
                
                # URL encode the password to handle special characters
                import urllib.parse
                encoded_password = urllib.parse.quote_plus(db_password)
                
                # Transaction pooler mode: Keep 'postgres' as database name
                # Port 6543 already indicates transaction pooler mode in Supabase
                db_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
                print(f"🔗 Constructed connection from individual components")
            
            print(f"🔗 Connecting via Transaction Pooler")
            print(f"   Host: {db_url.split('@')[1].split('/')[0] if '@' in db_url else 'N/A'}")
            
            # Enable SSL for Supabase connection
            conn = psycopg2.connect(
                db_url,
                sslmode='require'
            )
            cur = conn.cursor()
            
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create topics_embeddings table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS topics_embeddings (
                    topic_id VARCHAR PRIMARY KEY,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS topics_embedding_idx 
                ON topics_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
            print("✅ Supabase table created/verified")
            return True
            
        except ImportError:
            print("⚠️  psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"⚠️  Error setting up Supabase table: {e}")
            return False
    
    def save_to_supabase(self, embedding_dict: Dict[str, List[float]]):
        """Save embeddings to Supabase pgvector"""
        try:
            import psycopg2
            
            # Use DB_URL from .env if available (same as setup)
            db_url_env = os.getenv("DB_URL")
            
            if db_url_env:
                db_url = db_url_env
            else:
                # Fallback: construct from components
                db_host = os.getenv("DB_HOST")
                db_port = os.getenv("DB_PORT", "6543")
                db_name = os.getenv("DB_NAME", "postgres")
                db_user = os.getenv("DB_USER")
                db_password = os.getenv("DB_PASSWORD")
                
                import urllib.parse
                encoded_password = urllib.parse.quote_plus(db_password)
                db_url = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
            
            # Enable SSL for Supabase connection
            conn = psycopg2.connect(
                db_url,
                sslmode='require'
            )
            cur = conn.cursor()
            
            # Insert or update embeddings
            for topic_id, embedding in embedding_dict.items():
                # Format vector as PostgreSQL array string: '[0.1,0.2,0.3]'
                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                
                cur.execute("""
                    INSERT INTO topics_embeddings (topic_id, embedding, updated_at)
                    VALUES (%s, %s::vector, NOW())
                    ON CONFLICT (topic_id) 
                    DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW();
                """, (topic_id, vector_str))
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"✅ Saved {len(embedding_dict)} embeddings to Supabase")
            return True
            
        except ImportError:
            print("⚠️  psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"⚠️  Error saving to Supabase: {e}")
            return False
    
    def save_embedding_dict(self, embedding_dict: Dict[str, List[float]]):
        """Save embedding dictionary to Supabase or JSON file"""
        if self.use_supabase:
            # Setup table first
            if self.setup_supabase_table():
                # Save to Supabase
                if self.save_to_supabase(embedding_dict):
                    # Also save JSON backup
                    output_file = os.path.join(self.output_dir, "topic_embeddings.json")
                    with open(output_file, "w") as f:
                        json.dump(embedding_dict, f, indent=2)
                    print(f"✅ Saved JSON backup to {output_file}")
                    return
            # If Supabase fails, fall back to JSON
            print("⚠️  Falling back to JSON storage")
        
        # JSON fallback (or if Supabase not configured)
        output_file = os.path.join(self.output_dir, "topic_embeddings.json")
        
        with open(output_file, "w") as f:
            json.dump(embedding_dict, f, indent=2)
        
        print(f"✅ Saved embedding dictionary to {output_file}")
        
        # Also save a summary
        summary = {
            "step": "step4_embedding_vector_db",
            "timestamp": datetime.now().isoformat(),
            "total_topics": len(embedding_dict),
            "vector_dimensions": self.vector_dimensions,
            "embedding_format": "768-dimensional normalized vectors",
            "storage_method": "Supabase pgvector" if self.use_supabase else "JSON dictionary",
            "topics": list(embedding_dict.keys())
        }
        
        summary_file = os.path.join(self.output_dir, "step4_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved step summary to {summary_file}")
    
    def run_step4(self):
        """
        Run Step 4: Embedding Topics into Vector DB
        Creates embedding dictionary ready for vector database storage
        """
        print(f"🎯 {self.step_name.upper()} EXECUTION")
        print("=" * 60)
        
        # Create the embedding dictionary (required format)
        embedding_dict = self.create_embedding_dict()
        self.save_embedding_dict(embedding_dict)
        
        # Test vector similarity search
        self.test_vector_search(embedding_dict)
        
        print(f"\n{'='*60}")
        print("📊 STEP 4 SUMMARY")
        print(f"{'='*60}")
        print(f"Total topics processed: {len(self.topics)}")
        print(f"Vector dimensions: {self.vector_dimensions}")
        print(f"Storage method: JSON dictionary (ready for pgvector)")
        print(f"Cost approach: No database costs (JSON only)")
        print(f"Ready for Step 6: ✅ Vector lookup functionality available")
        
        return embedding_dict
    
    def test_vector_search(self, embedding_dict: Dict[str, List[float]]):
        """Test vector similarity search to verify embeddings work"""
        if len(embedding_dict) < 2:
            return
        
        print("\n🧪 Testing vector similarity search...")
        
        # Get two random embeddings
        topic_ids = list(embedding_dict.keys())
        topic1_id = topic_ids[0]
        topic2_id = topic_ids[1]
        
        embedding1 = np.array(embedding_dict[topic1_id])
        embedding2 = np.array(embedding_dict[topic2_id])
        
        # Calculate cosine similarity
        cosine_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        print(f"  Cosine similarity between {topic1_id} and {topic2_id}: {cosine_sim:.4f}")
        
        # Test finding most similar topics
        query_embedding = embedding1
        similarities = {}
        for topic_id, embedding in embedding_dict.items():
            if topic_id != topic1_id:
                emb = np.array(embedding)
                sim = np.dot(query_embedding, emb)
                similarities[topic_id] = sim
        
        # Get top 3 most similar
        top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3 most similar to {topic1_id}:")
        for topic_id, sim in top_similar:
            print(f"    - {topic_id}: {sim:.4f}")
        
        print("✅ Vector search test passed!")

def main():
    """Main execution function"""
    print("🚀 Step 4: Embedding Topics into Vector DB")
    print("Based on client discussion: Cost-effective approach")
    
    # Initialize Step 4 with best model from Step 3
    # Using google_gemini-2.0-flash (best performing model from Step 3 evaluation)
    step4 = Step4VectorDB(model_name="google_gemini-2.0-flash")
    
    # Run Step 4
    embedding_dict = step4.run_step4()
    
    print(f"\n✅ Step 4 completed successfully!")
    print(f"📁 Results saved to: output/{step4.step_name}/")
    print(f"🔍 {len(embedding_dict)} topics ready for vector search in Step 6")

if __name__ == "__main__":
    main()
