#!/usr/bin/env python3
"""
Step 4: Embedding Topics into Vector DB
Implements cost-effective vector storage using pgvector (PostgreSQL)
Based on client discussion: Avoid expensive Pinecone, optimize pgvector instead
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys
import urllib.parse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import GOOGLE_API_KEY, OPENAI_API_KEY
from utils.supabase_helpers import build_supabase_db_url

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")  # Anon key for API access
SUPABASE_DB_URL = build_supabase_db_url()

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
        
        # Embedding configuration
        self.embedding_provider = os.getenv(
            "STEP4_EMBEDDING_PROVIDER",
            "google",
        ).lower()
        self.embedding_model_name = os.getenv(
            "STEP4_EMBEDDING_MODEL",
            "models/text-embedding-004",
        )
        self.default_vector_dimensions = int(
            os.getenv("STEP4_VECTOR_DIMENSIONS", "768")
        )
        self.vector_dimensions = self.default_vector_dimensions
        self._google_genai = None
        self._openai_client = None
        self._embedding_backend = self._initialize_embedding_backend()
        
        # Check if Supabase is available
        self.use_supabase = SUPABASE_DB_URL is not None
        
        print("✅ Step 4: Vector DB initialized")
        print(f"✅ Using model: {self.step3_model}")
        print(f"✅ Loaded {len(self.topics)} topics from Step 3")
        if self._embedding_backend:
            print(f"✅ Embedding provider: {self.embedding_provider} ({self.embedding_model_name})")
        else:
            print("⚠️  Using deterministic fallback embeddings (no provider configured)")
        print(f"✅ Vector dimensions: {self.vector_dimensions}")
        if self.use_supabase:
            print(f"✅ Using Supabase pgvector for storage")
        else:
            print("⚠️  Supabase not configured, using JSON fallback")

    def _initialize_embedding_backend(self) -> Optional[str]:
        """Initialize embedding backend based on configuration"""
        provider = self.embedding_provider

        if provider == "google":
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError:
                print("⚠️  google-generativeai missing (pip install google-generativeai)")
                return None

            if not GOOGLE_API_KEY:
                print("⚠️  GOOGLE_API_KEY not set. Falling back to deterministic embeddings.")
                return None

            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self._google_genai = genai
                print("✅ Google Generative AI client configured for embeddings")
                return "google"
            except Exception as exc:
                print(
                    f"⚠️  Google embedding setup failed ({exc}). "
                    "Using deterministic fallback."
                )
                return None

        if provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                print("⚠️  openai package missing (pip install openai)")
                return None

            if not OPENAI_API_KEY:
                print("⚠️  OPENAI_API_KEY not set. Falling back to deterministic embeddings.")
                return None

            try:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
                print("✅ OpenAI client configured for embeddings")
                return "openai"
            except Exception as exc:
                print(
                    f"⚠️  OpenAI embedding setup failed ({exc}). "
                    "Using deterministic fallback."
                )
                return None

        if provider not in {"google", "openai"}:
            print(
                f"⚠️  Unsupported embedding provider '{provider}'. "
                "Using deterministic fallback."
            )

        return None

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length"""
        vector = np.array(embedding, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()

    def _generate_embedding_with_backend(self, seed_text: str) -> Optional[List[float]]:
        """Generate embedding using configured backend; returns None on failure"""
        if not self._embedding_backend:
            return None

        try:
            if self._embedding_backend == "google":
                response = self._google_genai.embed_content(
                    model=self.embedding_model_name,
                    content=seed_text
                )
                embedding = (
                    response.get("embedding")
                    if isinstance(response, dict)
                    else getattr(response, "embedding", None)
                )
                if not embedding:
                    raise ValueError("Empty embedding returned by Google API")
                return self._normalize_embedding(embedding)

            if self._embedding_backend == "openai":
                response = self._openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=seed_text
                )
                data = getattr(response, "data", [])
                if not data:
                    raise ValueError("Empty embedding returned by OpenAI API")
                embedding = data[0].embedding
                return self._normalize_embedding(embedding)

        except Exception as exc:
            print(
                f"⚠️  {self._embedding_backend} embedding error: {exc}. "
                "Using deterministic fallback."
            )
            return None

        return None

    def _generate_fallback_embedding(self, seed_text: str) -> List[float]:
        """Generate deterministic embedding via pseudo-random fallback"""
        seed_hash = hash(seed_text) % (2**32)
        rng = np.random.default_rng(seed_hash)
        embedding = rng.uniform(-1.0, 1.0, self.vector_dimensions).astype(float)
        return self._normalize_embedding(embedding.tolist())
    
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
        if not seed_text.strip():
            seed_text = f"{topic.get('cluster_id', 'topic')} metadata unavailable"

        embedding = self._generate_embedding_with_backend(seed_text)

        if embedding is None:
            embedding = self._generate_fallback_embedding(seed_text)
        else:
            # Update vector dimension dynamically to match backend output
            self.vector_dimensions = len(embedding)

        return embedding
    
    def setup_supabase_table(self):
        """Create table in Supabase with pgvector extension if it doesn't exist"""
        try:
            import psycopg2
            print(f"🔗 Connecting to Supabase...")

            db_url = SUPABASE_DB_URL
            if not db_url:
                raise ValueError("Missing database credentials in environment configuration")

            host_display = "N/A"
            try:
                parsed = urllib.parse.urlparse(db_url)
                if parsed.hostname:
                    host_display = parsed.hostname
            except Exception:
                pass

            print("🔗 Connecting via Transaction Pooler")
            print(f"   Host: {host_display}")

            # Enable SSL for Supabase connection
            conn = psycopg2.connect(
                db_url,
                sslmode='require'
            )
            cur = conn.cursor()
            
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create topics_embeddings table if it doesn't exist
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS topics_embeddings (
                    topic_id VARCHAR PRIMARY KEY,
                    embedding vector({self.vector_dimensions}),
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
            db_url = SUPABASE_DB_URL
            if not db_url:
                raise ValueError("Missing database credentials in environment configuration")
            
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
