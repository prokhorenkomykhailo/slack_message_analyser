# Step 4: Embedding Topics into Vector Database

## Overview
Step 4 converts enriched topics from Step 3 into vector embeddings and stores them in a vector database (Supabase pgvector) for fast semantic search.

## Input
- **Source**: Step 3 metadata output (`output/step3_client_analysis/google_gemini-2.0-flash.json`)
- **Content**: Enriched topic metadata (title, summary, action items, participants, tags, urgency, etc.)

## Processing Steps

### 1. Load Topics from Step 3
- Loads `metadata_results` from Step 3 JSON file for the specified model (default: `google_gemini-2.0-flash`)
- Extracts all enriched topics with their complete metadata

### 2. Generate Embeddings
For each topic, creates a **768-dimensional embedding vector**:
- **Method**: Deterministic embedding generation based on topic metadata
- **Inputs used**:
  - Title
  - Summary
  - Tags
  - Participants
  - Urgency
- **Process**:
  - Combines all metadata fields into a seed text
  - Generates hash from seed text for deterministic results
  - Creates 768-dimensional vector with normalized values (-1 to 1)
  - Normalizes vector (required for cosine similarity in pgvector)
- **Result**: `{topic_id: [768-dimensional vector]}`

### 3. Store in Vector Database
**Primary Method: Supabase pgvector**
- **Connection**: Uses transaction pooler mode (port 6543) with SSL
- **Table Creation**: 
  - Creates `topics_embeddings` table if not exists
  - Enables `pgvector` extension
  - Creates index: `ivfflat` with cosine similarity operator
- **Storage**:
  - Inserts or updates embeddings in PostgreSQL
  - Each row: `topic_id` (VARCHAR), `embedding` (vector(768)), timestamps
- **Index**: IVFFlat index for fast similarity search

**Fallback Method: JSON File**
- If Supabase is unavailable or not configured
- Saves `topic_embeddings.json` with embedding dictionary

### 4. Verify Embeddings
Tests vector similarity search:
- Calculates cosine similarity between embeddings
- Finds top 3 most similar topics
- Verifies embeddings are properly normalized and functional

## Output
1. **Supabase Database** (if configured):
   - Table: `topics_embeddings`
   - Indexed for fast similarity search
   - Ready for Step 6 semantic search

2. **JSON Files** (always created):
   - `output/step4_embedding_vector_db/topic_embeddings.json`: Full embedding dictionary
   - `output/step4_embedding_vector_db/step4_summary.json`: Summary with metadata

## Purpose
- **Enable Semantic Search**: Topics can be found by similarity to new messages
- **Fast Lookup**: Vector database allows efficient similarity queries
- **Cost-Effective**: Uses Supabase pgvector (included) instead of expensive services like Pinecone
- **Ready for Step 6**: New messages can be embedded and matched to existing topics

## Key Features
- ✅ Deterministic embeddings (same topic = same vector)
- ✅ Normalized vectors (for accurate cosine similarity)
- ✅ Indexed for fast search (IVFFlat index)
- ✅ Fallback to JSON if database unavailable
- ✅ Vector similarity testing included

## Technical Details
- **Vector Dimensions**: 768 (standard embedding size)
- **Similarity Metric**: Cosine similarity
- **Database**: PostgreSQL with pgvector extension
- **Index Type**: IVFFlat with 100 lists
- **Connection**: Transaction pooler mode with SSL required
