#!/usr/bin/env python3
"""
Step 6: New Message Processing
When a new message arrives, determine if it belongs to an existing topic or needs a new topic.
Uses Step 4 vector embeddings for similarity search and Step 3 AI for decision making.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import GOOGLE_API_KEY, OPENAI_API_KEY
from utils.supabase_helpers import build_supabase_db_url
from utils.model_clients import call_model_with_retry

SUPABASE_DB_URL = build_supabase_db_url()

class Step6NewMessageProcessing:
    """Step 6: Process new messages and update/create topics"""
    
    def __init__(self):
        self.step_name = "step6_new_message_processing"
        self.output_dir = os.path.join("output", self.step_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Supabase connection info
        self.supabase_db_url = SUPABASE_DB_URL
        self.use_supabase = self.supabase_db_url is not None
        
        # Use the same model as Step 3 for consistency
        self.metadata_model = "google"
        self.metadata_model_name = "gemini-2.0-flash"
        
        # Load existing topics from Step 3
        self.existing_topics = self.load_topics_from_step3()
        
        # Load embeddings from Step 4 / Supabase
        self.topic_embeddings = self.load_embeddings_from_step4()
        self.total_embeddings = len(self.topic_embeddings)

        # Embedding configuration (mirror Step 4 defaults)
        default_provider = os.getenv("STEP4_EMBEDDING_PROVIDER", "google")
        default_model = os.getenv("STEP4_EMBEDDING_MODEL", "models/text-embedding-004")
        default_dim = os.getenv("STEP4_VECTOR_DIMENSIONS", "768")

        self.embedding_provider = os.getenv(
            "STEP6_EMBEDDING_PROVIDER", default_provider
        ).lower()
        self.embedding_model_name = os.getenv(
            "STEP6_EMBEDDING_MODEL", default_model
        )
        self.default_vector_dimensions = int(
            os.getenv("STEP6_VECTOR_DIMENSIONS", default_dim)
        )
        self.vector_dimensions = self.default_vector_dimensions
        self._google_genai = None
        self._openai_client = None
        self._embedding_backend = self._initialize_embedding_backend()

        # Align vector dimensions with Step 4 embeddings if available
        if self.topic_embeddings:
            sample_vector = next(iter(self.topic_embeddings.values()))
            if isinstance(sample_vector, list):
                self.vector_dimensions = len(sample_vector)
        
        print(f"✅ Step 6: New Message Processing initialized")
        print(f"✅ Loaded {len(self.existing_topics)} existing topics")
        if self.use_supabase:
            print(f"✅ Supabase pgvector enabled")
        print(f"✅ Loaded {self.total_embeddings} topic embeddings")
        if self._embedding_backend:
            print(
                f"✅ Message embedding provider: "
                f"{self.embedding_provider} ({self.embedding_model_name})"
            )
        else:
            print(
                "⚠️  Message embeddings using deterministic fallback "
                "(no provider configured)"
            )

    def _initialize_embedding_backend(self) -> Optional[str]:
        """Configure embedding backend for new messages."""
        provider = self.embedding_provider

        if provider == "google":
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError:
                print(
                    "⚠️  google-generativeai missing "
                    "(pip install google-generativeai)"
                )
                return None

            if not GOOGLE_API_KEY:
                print(
                    "⚠️  GOOGLE_API_KEY not set. "
                    "Using deterministic message embeddings."
                )
                return None

            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self._google_genai = genai
                return "google"
            except Exception as exc:
                print(
                    f"⚠️  Google embedding setup failed ({exc}). "
                    "Using deterministic message embeddings."
                )
                return None

        if provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                print("⚠️  openai package missing (pip install openai)")
                return None

            if not OPENAI_API_KEY:
                print(
                    "⚠️  OPENAI_API_KEY not set. "
                    "Using deterministic message embeddings."
                )
                return None

            try:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
                return "openai"
            except Exception as exc:
                print(
                    f"⚠️  OpenAI embedding setup failed ({exc}). "
                    "Using deterministic message embeddings."
                )
                return None

        if provider not in {"google", "openai"}:
            print(
                f"⚠️  Unsupported embedding provider '{provider}'. "
                "Using deterministic message embeddings."
            )

        return None

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        vector = np.array(embedding, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.tolist()
        return (vector / norm).tolist()

    def _generate_embedding_with_backend(self, text: str) -> Optional[List[float]]:
        """Generate embedding for message using configured backend."""
        if not self._embedding_backend:
            return None

        try:
            if self._embedding_backend == "google":
                response = self._google_genai.embed_content(
                    model=self.embedding_model_name,
                    content=text,
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
                    input=text,
                )
                data = getattr(response, "data", [])
                if not data:
                    raise ValueError("Empty embedding returned by OpenAI API")
                embedding = data[0].embedding
                return self._normalize_embedding(embedding)

        except Exception as exc:
            print(
                f"⚠️  {self._embedding_backend} embedding error: {exc}. "
                "Using deterministic message embeddings."
            )
            return None

        return None

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate deterministic fallback embedding matching configured size."""
        seed_hash = hash(text) % (2**32)
        rng = np.random.default_rng(seed_hash)
        embedding = rng.normal(0.0, 0.1, self.vector_dimensions).astype(float)
        return self._normalize_embedding(embedding.tolist())
    
    def load_topics_from_step3(self) -> List[Dict]:
        """Load topics from Step 3 JSON file"""
        try:
            candidate_files = [
                os.path.join(
                    "output",
                    "step3_client_analysis",
                    f"{self.metadata_model_name}.json",
                ),
                os.path.join(
                    "output",
                    "phase5_metadata_generation",
                    f"{self.metadata_model}_{self.metadata_model_name}.json",
                ),
                os.path.join(
                    "output",
                    "phase5_metadata_generation",
                    f"{self.metadata_model_name}.json",
                ),
            ]
            
            step3_file = next((path for path in candidate_files if os.path.exists(path)), None)
            
            if not step3_file:
                print("❌ Step 3 metadata file not found in expected locations:")
                for candidate in candidate_files:
                    print(f"   - {candidate}")
                return []
            
            with open(step3_file, "r") as f:
                step3_data = json.load(f)
            
            if "metadata_results" in step3_data:
                topics = step3_data["metadata_results"]
                print(f"✅ Loaded {len(topics)} topics from Step 3")
                print(f"   📂 Source file: {step3_file}")
                return topics
            else:
                print(f"❌ No 'metadata_results' found in Step 3 file")
                return []
            
        except Exception as e:
            print(f"❌ Error loading topics: {e}")
            return []
    
    def load_embeddings_from_step4(self) -> Dict[str, List[float]]:
        """Load embeddings from Step 4 JSON file"""
        try:
            if self.use_supabase:
                embeddings = self.load_embeddings_from_supabase()
                if embeddings:
                    print(f"✅ Loaded {len(embeddings)} embeddings from Supabase")
                    return embeddings
                else:
                    print("⚠️  Supabase embeddings could not be loaded, falling back to JSON")
            
            step4_file = os.path.join(
                "output",
                "step4_embedding_vector_db",
                "topic_embeddings.json",
            )
            
            if not os.path.exists(step4_file):
                print(f"⚠️  Step 4 embeddings file not found: {step4_file}")
                print("⚠️  Will generate embeddings on-the-fly")
                return {}
            
            with open(step4_file, "r") as f:
                embeddings = json.load(f)
            
            print(f"✅ Loaded {len(embeddings)} embeddings from Step 4")
            return embeddings
            
        except Exception as e:
            print(f"⚠️  Error loading embeddings: {e}")
            return {}
    
    def load_embeddings_from_supabase(self) -> Dict[str, List[float]]:
        """Fetch embeddings stored in Supabase pgvector."""
        try:
            import psycopg2
        except ImportError:
            print("⚠️  psycopg2 not installed. Install with: pip install psycopg2-binary")
            return {}
        
        if not self.supabase_db_url:
            return {}
        
        try:
            conn = psycopg2.connect(
                self.supabase_db_url,
                sslmode="require",
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT topic_id, embedding FROM topics_embeddings;"
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"⚠️  Error fetching embeddings from Supabase: {e}")
            return {}
        
        embeddings: Dict[str, List[float]] = {}
        for topic_id, embedding in rows:
            vector: List[float]
            if isinstance(embedding, (list, tuple)):
                vector = [float(x) for x in embedding]
            elif hasattr(embedding, "tolist"):
                vector = [float(x) for x in embedding.tolist()]
            else:
                try:
                    cleaned = str(embedding).strip("[]")
                    vector = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
                except Exception:
                    continue
            
            if vector:
                embeddings[str(topic_id)] = vector
        
        return embeddings
    
    def _extract_mentions(self, text: str) -> str:
        """Extract @mentions from message text (the 'to' field)"""
        import re
        mentions = re.findall(r'@\w+', text)
        return ' '.join(mentions) if mentions else ''
    
    def generate_message_embedding(self, message: Dict) -> List[float]:
        """
        Generate embedding for a new message.
        Includes channel, thread_id, from (sender), and to (mentions) for similarity matching.
        Matches the format used in Step 4 topic embeddings.
        """
        # Extract message fields
        text = message.get("text", "")
        user = message.get("user", "")  # "from" - the sender
        channel = message.get("channel", "")
        thread_id = message.get("thread_ts") or message.get("thread_id", "")
        
        # Extract mentions from text (the "to" field - who the message is addressed to)
        mentions = self._extract_mentions(text)
        
        # Format thread_id similar to topic embedding (thread_root)
        thread_part = f"thread_{thread_id}" if thread_id else ""
        
        # Weight structural metadata by repeating them before embedding
        # This gives them stronger influence in the embedding space
        # Weights: channel=3x, thread=3x, from/to=2x, text=1x
        channel_weighted = f"{channel} {channel} {channel}" if channel else ""
        thread_weighted = f"{thread_part} {thread_part} {thread_part}" if thread_part else ""
        from_part = f"from_{user}" if user else ""
        to_part = f"to_{mentions}" if mentions else ""
        from_to_weighted = f"{from_part} {to_part} {from_part} {to_part}" if (from_part or to_part) else ""
        
        # Combine with weighted structural metadata first (higher influence)
        # Format: "channel*3 thread*3 from_to*2 text*1"
        combined_text = f"{channel_weighted} {thread_weighted} {from_to_weighted} {text}".strip().lower()
        if not combined_text:
            combined_text = (
                f"{channel} {from_part} empty message content"
            ).strip().lower()

        embedding = self._generate_embedding_with_backend(combined_text)
        if embedding is None:
            embedding = self._generate_fallback_embedding(combined_text)

        return embedding
    
    def find_similar_topics(self, message: Dict, message_embedding: List[float], threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find similar topics using cosine similarity of embeddings.
        
        Note: Structural metadata (channel, thread, from/to) is already weighted in embeddings
        by repeating them multiple times (channel 3x, thread 3x, from/to 2x).
        This means if threads match, similarity will naturally be high.
        No need for complex fallback rules or thread matching logic.
        
        Returns list of (topic_id, similarity_score) tuples.
        """
        if self.use_supabase and not self.topic_embeddings:
            # Attempt reloading before giving up
            self.topic_embeddings = self.load_embeddings_from_supabase()
            self.total_embeddings = len(self.topic_embeddings)
        
        if not self.topic_embeddings:
            return []
        
        similar_topics = []
        message_vec = np.array(message_embedding)
        
        for topic_id, topic_embedding in self.topic_embeddings.items():
            topic_vec = np.array(topic_embedding)
            
            # Calculate cosine similarity
            dot_product = np.dot(message_vec, topic_vec)
            norm_product = np.linalg.norm(message_vec) * np.linalg.norm(topic_vec)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                
                # Include if similarity meets threshold
                if similarity >= threshold:
                    similar_topics.append((topic_id, float(similarity)))
        
        # Sort by similarity score (highest first)
        similar_topics.sort(key=lambda x: x[1], reverse=True)
        return similar_topics
    
    def should_update_existing_topic(self, message: Dict, similar_topics: List[Tuple[str, float]]) -> Optional[str]:
        """
        Use AI to decide if message belongs to an existing topic.
        Returns topic_id if it should update, None if it should create new.
        """
        if not similar_topics:
            return None
        
        # Get top 3 similar topics
        top_topics = similar_topics[:3]
        
        # Format existing topics for prompt
        topics_info = []
        for topic_id, similarity in top_topics:
            topic = next((t for t in self.existing_topics if t.get("cluster_id") == topic_id), None)
            if topic and topic.get("metadata"):
                metadata = topic["metadata"]
                topics_info.append(f"""
Topic ID: {topic_id}
Title: {metadata.get("title", "N/A")}
Summary: {metadata.get("summary", "N/A")}
Channel: {metadata.get("channel", "N/A")}
Participants: {metadata.get("participants", [])}
Similarity Score: {similarity:.2f}
""")
        
        topics_str = "\n".join(topics_info)
        
        # Create decision prompt
        prompt = f"""
You are an expert at analyzing whether a new message belongs to an existing topic.

**New Message:**
Channel: {message.get("channel", "N/A")}
User: {message.get("user", "N/A")}
Text: {message.get("text", "")}
Timestamp: {message.get("timestamp", "N/A")}

**Similar Existing Topics (found via vector similarity):**
{topics_str}

**Task:**
Determine if the new message belongs to one of these existing topics or should create a new topic.

**Decision Criteria:**
1. Same project/client name mentioned
2. Same participants involved
3. Same channel and thread context
4. Continuation of the same discussion
5. Related action items or decisions

**Output Format (JSON only):**
{{
  "decision": "update_existing" or "create_new",
  "topic_id": "topic_001" or null,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}

Analyze and provide your decision in JSON format.
"""
        
        try:
            result = call_model_with_retry(
                self.metadata_model,
                self.metadata_model_name,
                prompt,
                max_retries=3
            )
            
            if not result.get("success"):
                print(f"⚠️  AI decision failed: {result.get('error', 'Unknown error')}")
                # Fallback: use highest similarity topic if similarity > 0.8
                if top_topics and top_topics[0][1] > 0.8:
                    return top_topics[0][0]
                return None
            
            # Parse JSON response
            response_text = result.get("response", "")
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                decision = json.loads(json_str)
                
                if decision.get("decision") == "update_existing" and decision.get("topic_id"):
                    print(f"✅ AI decision: Update existing topic {decision.get('topic_id')}")
                    return decision.get("topic_id")
                else:
                    print(f"✅ AI decision: Create new topic")
                    return None
            else:
                # Fallback
                if top_topics and top_topics[0][1] > 0.8:
                    return top_topics[0][0]
                return None
                
        except Exception as e:
            print(f"⚠️  Error in AI decision: {e}")
            # Fallback: use highest similarity if > 0.8
            if top_topics and top_topics[0][1] > 0.8:
                return top_topics[0][0]
            return None
    
    def is_message_topic_worthy(self, message: Dict) -> Dict:
        """
        Evaluate if a message contains enough substance to create a new topic.
        
        A message is topic-worthy if it contains:
        - Actionable information (tasks, deadlines, decisions, requests)
        - Important announcements (projects, policies, hiring, updates)
        - Substantial information exchange (not just casual questions/comments)
        - Enough context to generate meaningful metadata
        
        Returns:
        {
            "is_worthy": bool,
            "reason": str,
            "suggested_action": "create" | "discard"
        }
        """
        prompt = f"""
You are evaluating if a message contains enough substance to create a topic.

**Message to Evaluate:**
Channel: {message.get("channel", "N/A")}
User: {message.get("user", "N/A")}
Text: {message.get("text", "")}
Timestamp: {message.get("timestamp", "N/A")}

**Definition of a Topic-Worthy Message:**
A message should create a topic ONLY if it contains at least one of these:
1. **Actionable information**: Tasks, deadlines, action items, decisions, commitments
2. **Important announcements**: New projects, policy changes, hiring, major updates
3. **Substantial information exchange**: Detailed explanations, reports, data, analysis
4. **Requests with context**: Requests for work, approvals, resources (not just casual questions)

**NOT Topic-Worthy (should be DISCARDED):**
- Casual questions without context ("I heard about X, can someone share updates?")
- Simple acknowledgments ("Thanks!", "Got it", "OK")
- Small talk or chit-chat
- Vague references to existing topics without new information
- Messages that don't contain enough context to generate meaningful metadata

**Your Task:**
Evaluate if this message is topic-worthy and should create a new topic, or if it should be discarded.

Respond in JSON format:
{{
    "is_worthy": true/false,
    "reason": "Brief explanation of why this message is or isn't topic-worthy",
    "suggested_action": "create" or "discard"
}}
"""
        
        try:
            result = call_model_with_retry(
                self.metadata_model,
                self.metadata_model_name,
                prompt,
                max_retries=3
            )
            
            if not result.get("success"):
                print(f"⚠️  Topic-worthiness evaluation failed: {result.get('error', 'Unknown error')}")
                # Conservative fallback: assume it's worthy to avoid losing potentially important messages
                return {
                    "is_worthy": True,
                    "reason": "Evaluation failed, assuming worthy to be safe",
                    "suggested_action": "create"
                }
            
            # Parse JSON response
            response_text = result.get("response", "")
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                evaluation = json.loads(json_str)
                return evaluation
            else:
                # Fallback
                return {
                    "is_worthy": True,
                    "reason": "Could not parse response, assuming worthy to be safe",
                    "suggested_action": "create"
                }
                
        except Exception as e:
            print(f"⚠️  Error evaluating topic-worthiness: {e}")
            # Conservative fallback
            return {
                "is_worthy": True,
                "reason": f"Error during evaluation: {e}",
                "suggested_action": "create"
            }
    
    def update_existing_topic(self, topic_id: str, message: Dict) -> Dict:
        """
        Update an existing topic with the new message.
        Regenerates metadata using Step 3 approach.
        """
        # Find the topic
        topic = next((t for t in self.existing_topics if t.get("cluster_id") == topic_id), None)
        if not topic:
            return {"success": False, "error": f"Topic {topic_id} not found"}
        
        # Get existing metadata
        existing_metadata = topic.get("metadata", {})
        
        # Create update prompt for Step 3 model
        prompt = f"""
You are updating an existing topic with a new message. Regenerate the topic metadata to include the new message.

**Existing Topic Metadata:**
Title: {existing_metadata.get("title", "N/A")}
Summary: {existing_metadata.get("summary", "N/A")}
Channel: {existing_metadata.get("channel", "N/A")}
Participants: {existing_metadata.get("participants", [])}
Action Items: {json.dumps(existing_metadata.get("action_items", []), indent=2)}
Tags: {existing_metadata.get("tags", [])}
Urgency: {existing_metadata.get("urgency", "N/A")}

**New Message to Add:**
Channel: {message.get("channel", "N/A")}
User: {message.get("user", "N/A")}
Text: {message.get("text", "")}
Timestamp: {message.get("timestamp", "N/A")}

**Task:**
Update the topic metadata to incorporate the new message:

1. **Update Summary**: Include new information from the message

2. **Action Items - CRITICAL RULES**:
   a. **KEEP ALL EXISTING ACTION ITEMS** - Do not remove or regenerate them
   b. **UPDATE EXISTING ACTION ITEM STATUS**:
      - If the message mentions completing a task, find the matching existing action item and change its status to "completed"
      - If the message mentions starting work, change status to "in_progress"
      - Match by task description, owner, or context (e.g., "completed content" matches "Prepare content for review")
   c. **ADD NEW ACTION ITEMS**:
      - Extract ALL actionable instructions from the message (e.g., "update timelines", "review by end of week", "confirm wire transfer")
      - Create new action items for each instruction
      - If multiple people are mentioned for the same task, create SEPARATE DISTINCT tasks for each person (NOT duplicates with same description)
      - Example: If message says "@sam @jordan please review", create two distinct tasks:
        * "Review content draft - design perspective" (owner: "@sam")
        * "Review content draft - legal perspective" (owner: "@jordan")
      - OR if it's truly the same task for multiple people, create one task with a clear description and assign to the primary owner, mentioning others in the task description
   d. **NO DUPLICATES**: Before adding a new action item, check if a similar one already exists

3. **Update Participants**: Add new users if mentioned

4. **Update Urgency**: Change if the message indicates urgency change

5. **Update Title**: Keep if still relevant, update if topic shifted

6. **Update Deadline**: Change if the message mentions a deadline change

**Output Format (JSON only):**
{{
  "title": "Updated title",
  "summary": "Updated summary including new message context",
  "action_items": [
    {{
      "task": "Task description",
      "owner": "@user",
      "due_date": "YYYY-MM-DD",
      "priority": "high/medium/low",
      "status": "pending/in_progress/completed"
    }}
  ],
  "participants": ["@user1", "@user2"],
  "urgency": "high/medium/low",
  "deadline": "YYYY-MM-DD",
  "status": "active/completed/pending/in_progress",
  "channel": "#channel-name",
  "tags": ["tag1", "tag2"]
}}

**CRITICAL ACTION ITEM RULES**:
1. Include ALL existing action items (preserve them exactly, but update status if completion is mentioned)
2. Extract EVERY actionable instruction from the new message and create action items
3. Update status of existing action items when the message indicates completion/progress
4. NO duplicate action items - if multiple people need to do the same task, create distinct tasks with different perspectives
5. Match existing action items to message content to update their status appropriately

**EXAMPLES**:
- Message: "Please update your timelines accordingly" → Create action item: "Update timelines to reflect new deadline"
- Message: "I've completed the first draft" → Find existing action item "Prepare content for review" and change status to "completed"
- Message: "@sam @jordan please review by end of week" → Create two distinct tasks: "Review content draft - design" (owner: "@sam") and "Review content draft - legal" (owner: "@jordan")

Provide the updated metadata in JSON format.
"""
        
        try:
            result = call_model_with_retry(
                self.metadata_model,
                self.metadata_model_name,
                prompt,
                max_retries=3
            )
            
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Unknown error")}
            
            # Parse response
            response_text = result.get("response", "")
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                updated_metadata = json.loads(json_str)
                
                return {
                    "success": True,
                    "topic_id": topic_id,
                    "action": "updated",
                    "metadata": updated_metadata
                }
            else:
                return {"success": False, "error": "Could not parse AI response"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_new_topic(self, message: Dict) -> Dict:
        """
        Create a new topic for the message.
        Uses Step 3 metadata generation approach.
        """
        # Create new topic prompt (similar to Step 3)
        prompt = f"""
You are creating a new topic from a single message. Generate comprehensive metadata.

**Message:**
Channel: {message.get("channel", "N/A")}
User: {message.get("user", "N/A")}
Text: {message.get("text", "")}
Timestamp: {message.get("timestamp", "N/A")}

**Task:**
Generate complete topic metadata including:
1. Descriptive title
2. Summary
3. Action items (if any)
4. Participants
5. Urgency
6. Tags
7. Channel

**Output Format (JSON only):**
{{
  "title": "Topic title",
  "summary": "Topic summary",
  "action_items": [
    {{
      "task": "Task description",
      "owner": "@user",
      "due_date": "YYYY-MM-DD",
      "priority": "high/medium/low",
      "status": "pending/in_progress/completed"
    }}
  ],
  "participants": ["@user1"],
  "urgency": "high/medium/low",
  "deadline": "YYYY-MM-DD",
  "status": "active",
  "channel": "#channel-name",
  "tags": ["tag1", "tag2"]
}}

Provide the metadata in JSON format.
"""
        
        try:
            result = call_model_with_retry(
                self.metadata_model,
                self.metadata_model_name,
                prompt,
                max_retries=3
            )
            
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Unknown error")}
            
            # Parse response
            response_text = result.get("response", "")
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                metadata = json.loads(json_str)
                
                # Generate new topic ID
                new_topic_id = f"topic_{len(self.existing_topics) + 1:03d}"
                
                return {
                    "success": True,
                    "topic_id": new_topic_id,
                    "action": "created",
                    "metadata": metadata
                }
            else:
                return {"success": False, "error": "Could not parse AI response"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_new_message(self, message: Dict) -> Dict:
        """
        Main function: Process a new message and decide to update or create topic.
        """
        print(f"\n🔄 Processing new message from {message.get('user', 'Unknown')} in {message.get('channel', 'Unknown')}")
        
        # Step 1: Generate embedding for new message
        message_embedding = self.generate_message_embedding(message)
        
        # Step 2: Find similar topics (with fallback rule for structural matches)
        similar_topics = self.find_similar_topics(message, message_embedding, threshold=0.7)
        
        if similar_topics:
            print(f"📊 Found {len(similar_topics)} similar topics (top similarity: {similar_topics[0][1]:.2f})")
        else:
            print(f"📊 No similar topics found (threshold: 0.7)")
        
        # Step 3: Use AI to decide
        topic_id = self.should_update_existing_topic(message, similar_topics)
        
        # Step 4: Update or create
        if topic_id:
            # Update existing topic
            result = self.update_existing_topic(topic_id, message)
            result["message"] = message
            result["similar_topics"] = similar_topics
            return result
        else:
            # No match found - check if message is topic-worthy before creating
            print(f"🔍 No matching topic found. Evaluating if message is topic-worthy...")
            worthiness = self.is_message_topic_worthy(message)
            
            if worthiness.get("is_worthy") and worthiness.get("suggested_action") == "create":
                print(f"✅ Message is topic-worthy: {worthiness.get('reason')}")
                result = self.create_new_topic(message)
                result["message"] = message
                result["similar_topics"] = similar_topics
                result["worthiness_evaluation"] = worthiness
                return result
            else:
                print(f"❌ Message is NOT topic-worthy: {worthiness.get('reason')}")
                print(f"🗑️  Message will be discarded (not creating topic)")
                return {
                    "success": True,
                    "action": "discarded",
                    "reason": worthiness.get("reason"),
                    "message": message,
                    "similar_topics": similar_topics,
                    "worthiness_evaluation": worthiness
                }
    
    def process_multiple_messages(self, messages: List[Dict]) -> Dict:
        """Process multiple new messages"""
        print(f"\n🎯 STEP 6: PROCESSING {len(messages)} NEW MESSAGES")
        print("=" * 60)
        
        results = []
        updated_topics = set()
        created_topics = []
        discarded_messages = []
        
        for i, message in enumerate(messages, 1):
            print(f"\n[{i}/{len(messages)}] Processing message...")
            result = self.process_new_message(message)
            results.append(result)
            
            if result.get("success"):
                if result.get("action") == "updated":
                    updated_topics.add(result.get("topic_id"))
                elif result.get("action") == "created":
                    created_topics.append(result.get("topic_id"))
                elif result.get("action") == "discarded":
                    discarded_messages.append({
                        "message_id": message.get("id"),
                        "user": message.get("user"),
                        "text": message.get("text"),
                        "reason": result.get("reason")
                    })
        
        # Save results
        output_file = os.path.join(self.output_dir, "processing_results.json")
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(messages),
                "updated_topics": list(updated_topics),
                "created_topics": created_topics,
                "discarded_messages": discarded_messages,
                "results": results
            }, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
        print(f"✅ Updated {len(updated_topics)} existing topics")
        print(f"✅ Created {len(created_topics)} new topics")
        print(f"🗑️  Discarded {len(discarded_messages)} messages (not topic-worthy)")
        
        return {
            "total_messages": len(messages),
            "updated_topics": list(updated_topics),
            "created_topics": created_topics,
            "discarded_messages": discarded_messages,
            "results": results
        }

def main():
    """Main execution function"""
    print("🚀 Step 6: New Message Processing")
    
    # Initialize Step 6
    step6 = Step6NewMessageProcessing()
    
    # Example: Process a new message
    # In production, this would come from Slack webhook or message queue
    # new_message = {
    #     "id": 301,
    #     "channel": "#campaign-briefs",
    #     "user": "Devon",
    #     "text": "@team The EcoBloom campaign deadline has been moved to August 5, 2025. Please update your timelines accordingly.",
    #     "timestamp": "2025-06-25T10:00:00"
    # }
    
    new_message = {
        "id": 301,
        "channel": "#finance-updates",
        "user": "Priya",
        "text": "@finance-team The Q3 invoice for Horizon Robotics (PO-HR-7784) is still unpaid. Please confirm that the wire transfer is scheduled before the July 3, 2025 cutoff to avoid penalties.",
        "timestamp": "2025-06-25T10:00:00"
    }
    # Process single message
    result = step6.process_new_message(new_message)
    
    print(f"\n✅ Step 6 completed!")
    print(f"📁 Results: {result}")

if __name__ == "__main__":
    main()

