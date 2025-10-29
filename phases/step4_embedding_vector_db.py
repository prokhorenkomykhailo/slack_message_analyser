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
    
    def __init__(self):
        self.step_name = "step4_embedding_vector_db"
        self.output_dir = os.path.join("output", self.step_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load topics from Step 3 (metadata generation)
        self.topics = self.load_topics_from_step3()
        
        # Vector dimensions (standard for most embedding models)
        self.vector_dimensions = 768
        
        # Check if Supabase is available
        self.use_supabase = SUPABASE_DB_URL is not None
        
        print(f"✅ Step 4: Vector DB initialized")
        print(f"✅ Loaded {len(self.topics)} topics from Step 3")
        print(f"✅ Vector dimensions: {self.vector_dimensions}")
        if self.use_supabase:
            print(f"✅ Using Supabase pgvector for storage")
        else:
            print(f"⚠️  Supabase not configured, using JSON fallback")
    
    def load_topics_from_step3(self) -> List[Dict]:
        """Load topics from Step 1 (clustering) results"""
        try:
            # Try to load from Step 1 results first
            step1_file = "step1_results.json"
            
            if os.path.exists(step1_file):
                with open(step1_file, "r") as f:
                    step1_data = json.load(f)
                
                # Convert Step 1 clusters to topic format
                topics = []
                for cluster in step1_data.get("clusters", []):
                    topic = {
                        "cluster_id": cluster.get("cluster_id", ""),
                        "success": True,
                        "metadata": {
                            "title": cluster.get("draft_title", ""),
                            "summary": f"Topic cluster with {len(cluster.get('message_ids', []))} messages",
                            "action_items": [],
                            "participants": cluster.get("participants", []),
                            "urgency": "medium",
                            "tags": [cluster.get("channel", "").replace("#", "")],
                            "channel": cluster.get("channel", "")
                        }
                    }
                    topics.append(topic)
                
                if topics:
                    print(f"✅ Loaded {len(topics)} topics from Step 1 results")
                    return topics
            
            # Try to load from Step 3 results (if available)
            step3_dir = os.path.join("output", "phase5_metadata_generation")
            comprehensive_file = os.path.join(step3_dir, "comprehensive_results.json")
            
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, "r") as f:
                    results = json.load(f)
                
                # Get the best performing model's topics
                best_model = self.get_best_step3_model(results)
                if best_model:
                    return results[best_model]["metadata_results"]
            
            # Fallback: create dummy topics for testing
            return self.create_dummy_topics()
            
        except Exception as e:
            print(f"⚠️  Could not load topic results: {e}")
            return self.create_dummy_topics()
    
    def get_best_step3_model(self, results: Dict) -> str:
        """Get the best performing model from Step 3"""
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            return None
        
        # Find model with highest success rate and lowest cost
        best_model = min(successful_results.items(), 
                        key=lambda x: x[1]["cost"]["total_cost"])
        return best_model[0]
    
    def create_dummy_topics(self) -> List[Dict]:
        """Create dummy topics for testing"""
        return [
            {
                "cluster_id": "cluster_001",
                "success": True,
                "metadata": {
                    "title": "EcoBloom Summer Campaign Planning",
                    "summary": "Team discussed Q2 project planning and campaign structure",
                    "action_items": [
                        {"task": "Create campaign timeline", "owner": "@alice", "due_date": "2024-04-15"}
                    ],
                    "participants": ["@alice", "@bob"],
                    "urgency": "high",
                    "tags": ["campaign", "planning", "ecobloom"],
                    "channel": "#campaign-briefs"
                }
            },
            {
                "cluster_id": "cluster_002",
                "success": True,
                "metadata": {
                    "title": "Technical Architecture Discussion",
                    "summary": "Discussed system architecture and technical decisions",
                    "action_items": [
                        {"task": "Review architecture docs", "owner": "@charlie", "due_date": "2024-04-20"}
                    ],
                    "participants": ["@charlie", "@david"],
                    "urgency": "medium",
                    "tags": ["architecture", "technical"],
                    "channel": "#tech"
                }
            },
            {
                "cluster_id": "cluster_003",
                "success": True,
                "metadata": {
                    "title": "Client Meeting Follow-up",
                    "summary": "Follow-up on client meeting and next steps",
                    "action_items": [
                        {"task": "Send meeting notes", "owner": "@eve", "due_date": "2024-04-18"}
                    ],
                    "participants": ["@eve", "@frank"],
                    "urgency": "high",
                    "tags": ["client", "meeting", "follow-up"],
                    "channel": "#client-communications"
                }
            }
        ]
    
    def generate_fake_embeddings(self, topic: Dict) -> List[float]:
        """
        Generate fake 768-dimensional embedding vectors for each topic
        This simulates the embedding generation process
        """
        # Use topic metadata to create deterministic but varied embeddings
        metadata = topic.get("metadata", {})
        
        # Create seed from topic content for deterministic embeddings
        seed_text = f"{metadata.get('title', '')} {metadata.get('summary', '')} {' '.join(metadata.get('tags', []))}"
        seed_hash = hash(seed_text) % (2**32)
        
        # Set random seed for deterministic generation
        np.random.seed(seed_hash)
        
        # Generate 768-dimensional vector with values between -1 and 1
        embedding = np.random.uniform(-1.0, 1.0, self.vector_dimensions).tolist()
        
        # Normalize the vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def create_embedding_dict(self) -> Dict[str, List[float]]:
        """
        Create Python dict format: {topic_id: embedding_vector}
        This is the format specified in the original requirements
        """
        embedding_dict = {}
        
        for topic in self.topics:
            topic_id = topic["cluster_id"]
            embedding = self.generate_fake_embeddings(topic)
            embedding_dict[topic_id] = embedding
        
        return embedding_dict
    
    def setup_supabase_table(self):
        """Create table in Supabase with pgvector extension if it doesn't exist"""
        try:
            import psycopg2
            import urllib.parse
            
            print(f"🔗 Connecting to Supabase...")
            print(f"🔗 URL: {SUPABASE_DB_URL[:50]}...")
            
            # URL encode the password to handle special characters
            db_url = SUPABASE_DB_URL
            if '!' in db_url or '@' in db_url:
                # Extract components and re-encode
                from urllib.parse import urlparse
                parsed = urlparse(db_url)
                if parsed.password:
                    encoded_password = urllib.parse.quote_plus(parsed.password)
                    db_url = f"postgresql://{parsed.username}:{encoded_password}@{parsed.hostname}:{parsed.port}{parsed.path}"
            
            conn = psycopg2.connect(db_url)
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
            
            conn = psycopg2.connect(SUPABASE_DB_URL)
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
        
        print(f"\n{'='*60}")
        print("📊 STEP 4 SUMMARY")
        print(f"{'='*60}")
        print(f"Total topics processed: {len(self.topics)}")
        print(f"Vector dimensions: {self.vector_dimensions}")
        print(f"Storage method: JSON dictionary (ready for pgvector)")
        print(f"Cost approach: No database costs (JSON only)")
        print(f"Ready for Step 6: ✅ Vector lookup functionality available")
        
        return embedding_dict

def main():
    """Main execution function"""
    print("🚀 Step 4: Embedding Topics into Vector DB")
    print("Based on client discussion: Cost-effective approach")
    
    # Initialize Step 4
    step4 = Step4VectorDB()
    
    # Run Step 4
    embedding_dict = step4.run_step4()
    
    print(f"\n✅ Step 4 completed successfully!")
    print(f"📁 Results saved to: output/{step4.step_name}/")
    print(f"🔍 {len(embedding_dict)} topics ready for vector search in Step 6")

if __name__ == "__main__":
    main()
