import json
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NLPProcessor:
    """
    NLP processor for understanding voice/text commands and matching candidates
    Uses sentence transformers for semantic understanding
    """
    
    def __init__(self):
        # Using a lightweight but effective model for semantic understanding
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.candidates = self.load_candidates()
        
    def load_candidates(self) -> List[Dict]:
        """Load resources from JSON file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, 'resources_data.json')
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading resources: {e}")
            return []
    
    def parse_command(self, command: str) -> Dict:
        """
        Parse natural language command to extract intent and parameters
        Supports commands like:
        - "Find me a React developer who is available"
        - "I need a senior Python engineer"
        - "Show me backend developers with AWS experience"
        """
        command_lower = command.lower()
        parsed = {
            'skills': [],
            'availability': None,
            'expertise_level': None,
            'specializations': []
        }
        
        # Extract skills
        common_skills = [
            'react', 'python', 'javascript', 'typescript', 'node.js', 'java', 'c#',
            'angular', 'vue.js', 'django', 'flask', 'fastapi', 'aws', 'azure', 'gcp',
            'docker', 'kubernetes', 'mongodb', 'postgresql', 'mysql', 'redis',
            'graphql', 'rest', 'api', 'microservices', 'machine learning', 'ml',
            'ai', 'nlp', 'data science', 'devops', 'frontend', 'backend', 'full stack',
            'mobile', 'ios', 'android', 'flutter', 'react native'
        ]
        
        for skill in common_skills:
            if skill in command_lower:
                parsed['skills'].append(skill)
        
        # Extract availability
        if any(word in command_lower for word in ['available', 'free', 'not busy']):
            parsed['availability'] = 'available'
        
        # Extract expertise level
        if 'senior' in command_lower or 'experienced' in command_lower:
            parsed['expertise_level'] = 'senior'
        elif 'junior' in command_lower or 'entry' in command_lower:
            parsed['expertise_level'] = 'junior'
        elif 'mid' in command_lower or 'intermediate' in command_lower:
            parsed['expertise_level'] = 'mid'
        elif 'expert' in command_lower or 'architect' in command_lower:
            parsed['expertise_level'] = 'expert'
        
        return parsed
    
    def semantic_search(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search using sentence transformers
        This allows for more flexible, meaning-based matching
        """
        # Create candidate descriptions
        candidate_texts = []
        for candidate in candidates:
            text = f"{candidate['title']} with {candidate['experience_years']} years experience. "
            text += f"Skills: {', '.join(candidate['skills'])}. "
            text += f"Specializations: {', '.join(candidate['specializations'])}. "
            text += f"Expertise: {candidate['expertise_level']}"
            candidate_texts.append(text)
        
        # Encode query and candidate descriptions
        query_embedding = self.model.encode([query])
        candidate_embeddings = self.model.encode(candidate_texts)
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top k candidates
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            candidate = candidates[idx].copy()
            candidate['match_score'] = float(similarities[idx])
            results.append(candidate)
        
        return results
    
    def filter_candidates(self, parsed_query: Dict) -> List[Dict]:
        """
        Filter candidates based on parsed query parameters
        """
        filtered = self.candidates.copy()
        
        # Filter by availability
        if parsed_query.get('availability'):
            filtered = [c for c in filtered if c['availability'] == parsed_query['availability']]
        
        # Filter by expertise level
        if parsed_query.get('expertise_level'):
            filtered = [c for c in filtered if c['expertise_level'] == parsed_query['expertise_level']]
        
        # Filter by skills (at least one matching skill)
        if parsed_query.get('skills'):
            filtered = [
                c for c in filtered 
                if any(
                    skill.lower() in [s.lower() for s in c['skills']] 
                    for skill in parsed_query['skills']
                )
            ]
        
        return filtered
    
    def search_candidates(self, command: str, top_k: int = 10) -> List[Dict]:
        """
        Main search function that combines parsing, filtering, and semantic search
        """
        # Parse the command
        parsed = self.parse_command(command)
        
        # First filter by hard constraints
        filtered_candidates = self.filter_candidates(parsed)
        
        # If we have filtered candidates, do semantic search on them
        # Otherwise, search all candidates
        if filtered_candidates:
            results = self.semantic_search(command, filtered_candidates, top_k)
        else:
            results = self.semantic_search(command, self.candidates, top_k)
        
        return results
    
    def recommend_for_task(self, task_description: str, top_k: int = 5) -> List[Dict]:
        """
        Recommend candidates for a specific task based on task description
        Focuses on available candidates
        """
        # Only consider available candidates
        available_candidates = [c for c in self.candidates if c['availability'] == 'available']
        
        if not available_candidates:
            return []
        
        # Perform semantic search
        results = self.semantic_search(task_description, available_candidates, top_k)
        
        # Add recommendation reasoning
        for result in results:
            result['recommendation_reason'] = self._generate_recommendation_reason(result, task_description)
        
        return results
    
    def _generate_recommendation_reason(self, candidate: Dict, task: str) -> str:
        """Generate a human-readable recommendation reason"""
        reasons = []
        
        reasons.append(f"{candidate['experience_years']} years of experience")
        reasons.append(f"{candidate['expertise_level']} level expertise")
        
        if candidate.get('rating'):
            reasons.append(f"{candidate['rating']}/5.0 rating")
        
        if candidate.get('projects_completed'):
            reasons.append(f"{candidate['projects_completed']} projects completed")
        
        return " • ".join(reasons)


