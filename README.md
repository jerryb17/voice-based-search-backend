# AI Recruitment Backend

This is the Python backend for the AI-powered recruitment platform with voice command capabilities.

## Features

- **50 Fake Candidate Data**: Pre-populated with diverse software developer candidates
- **NLP-Powered Search**: Uses sentence transformers for semantic understanding of queries
- **Voice Command Ready**: API designed to accept text input from voice commands
- **Smart Recommendations**: Recommends available candidates based on task descriptions
- **RESTful API**: Clean API endpoints for frontend integration

## Tech Stack

- **Flask**: Lightweight web framework
- **Sentence Transformers**: For NLP and semantic search
- **Scikit-learn**: For similarity calculations
- **CORS Enabled**: For React frontend communication

## Setup

1. Install Python 3.8+ (if not already installed)

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### GET /api/health
Health check endpoint

### GET /api/candidates
Get all candidates with optional filtering
- Query params: `availability`, `skill`, `expertise_level`, `min_rating`

### GET /api/candidates/:id
Get specific candidate by ID

### POST /api/search
Search candidates using natural language query
```json
{
  "query": "Find me a senior React developer who is available",
  "top_k": 10
}
```

### POST /api/recommend
Get candidate recommendations for a task
```json
{
  "task_description": "Build a React dashboard with real-time data visualization",
  "top_k": 5
}
```

### GET /api/skills
Get all unique skills across candidates

### GET /api/stats
Get statistics about candidates

## NLP Features

The NLP processor (`nlp_processor.py`) provides:

1. **Command Parsing**: Extracts intent from natural language
   - Skills: "React", "Python", "AWS", etc.
   - Availability: "available", "free"
   - Expertise: "senior", "junior", "mid"

2. **Semantic Search**: Uses sentence transformers for meaning-based matching
   - Understands context and intent
   - Returns ranked results with match scores

3. **Smart Recommendations**: Recommends best available candidates
   - Considers task requirements
   - Provides reasoning for recommendations

## Example Queries

Voice/Text commands that work:
- "Find me a React developer who is available"
- "I need a senior Python engineer with AWS experience"
- "Show me backend developers"
- "Get me a machine learning expert"
- "Find available full stack developers"

## Data Structure

Each candidate has:
- Basic info (name, email, phone, location)
- Professional details (title, experience, expertise level)
- Skills array
- Availability status
- Hourly rate
- Rating and projects completed
- Specializations

