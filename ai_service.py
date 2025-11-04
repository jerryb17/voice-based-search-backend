"""
AI Service for intelligent task analysis and resource matching
Supports both Google Gemini and OpenAI GPT
"""
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIService:
    def __init__(self):
        self.ai_provider = os.getenv('AI_PROVIDER', 'gemini').lower()
        
        # Initialize Gemini
        if GEMINI_AVAILABLE and self.ai_provider in ['gemini', 'both']:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key and gemini_key != 'your_gemini_api_key_here':
                genai.configure(api_key=gemini_key)
                # Use gemini-2.0-flash (stable and fast model)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                print("✅ Gemini AI initialized (using gemini-2.0-flash)")
            else:
                self.gemini_model = None
                print("⚠️  Gemini API key not configured - using fallback analysis")
        else:
            self.gemini_model = None
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and self.ai_provider in ['openai', 'both']:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key and openai_key != 'your_openai_api_key_here':
                self.openai_client = OpenAI(api_key=openai_key)
                print("✅ OpenAI GPT initialized")
            else:
                self.openai_client = None
                print("⚠️  OpenAI API key not configured")
        else:
            self.openai_client = None
    
    def analyze_task(self, task_description: str, task_title: str = "") -> Dict:
        """
        Analyze a task using AI or fallback to keyword extraction
        Returns task analysis even if AI is not configured
        """
        """
        Analyze a task description and extract:
        - Required skills
        - Department
        - Complexity level
        - Priority
        - Estimated hours
        """
        prompt = f"""You are an expert technical recruiter and project analyzer. Analyze the following task/query and extract detailed requirements.

**CRITICAL SKILL MATCHING RULES:**
1. When "AND" is used (e.g., "React AND Python" or "React Python"), BOTH skills are REQUIRED
2. List all REQUIRED skills separately - do not combine them  
3. Set "all_skills_required" to true when ALL skills must be present (uses "and"/"&" or comma)
4. Set "all_skills_required" to false when ANY skill is acceptable (uses "or")

**SKILL RELATIONSHIPS (Treat as equivalent):**
- .NET = ASP.NET = C# .NET = DotNet = ASPNET
- Node.js = NodeJS = Node
- .NET = ASP.NET = .NET Core = dotnet
- Azure = Microsoft Azure = MS Azure
- AWS = Amazon Web Services
- GCP = Google Cloud Platform
- React = ReactJS = React.js
- Vue = VueJS = Vue.js  
- Angular = AngularJS = Angular.js
- Node = NodeJS = Node.js
- Python = Python3
- JavaScript = JS = ECMAScript
- TypeScript = TS
- Kubernetes = K8s
- C# = CSharp
- AI = Machine Learning = ML
- NLP = Natural Language Processing
- Chatbot = Chat Bot

**Task Title:** {task_title}
**Query/Description:** {task_description}

Extract and return JSON with these fields:
1. required_skills: List ALL required skills (if "A and B", list both ["A", "B"])
2. all_skills_required: true if ALL skills needed, false if ANY skill acceptable
3. related_skills: Related/nice-to-have skills
4. department: Engineering, Design, Marketing, Data, etc.
5. complexity: "low", "medium", "high", or "critical"  
6. priority: "low", "medium", "high", or "critical"
7. estimated_hours: 10-200
8. key_requirements: Brief summary

**EXAMPLES:**

Query: "Find me a React and Python developer"
{{
  "required_skills": ["React", "Python"],
  "all_skills_required": true,
  "related_skills": ["JavaScript", "TypeScript", "Django"],
  "department": "Engineering",
  "complexity": "medium",
  "priority": "medium",
  "estimated_hours": 40,
  "key_requirements": "Full stack developer with React and Python"
}}

Query: "Find me an Angular and .NET developer"  
{{
  "required_skills": ["Angular", ".NET"],
  "all_skills_required": true,
  "related_skills": ["TypeScript", "C#", "ASP.NET", ".NET Core"],
  "department": "Engineering",
  "complexity": "medium",
  "priority": "medium",
  "estimated_hours": 40,
  "key_requirements": "Full stack with Angular and .NET"
}}

Query: "Azure developer"
{{
  "required_skills": ["Azure"],
  "all_skills_required": true,
  "related_skills": ["Cloud", "DevOps", "Terraform", "Kubernetes"],
  "department": "DevOps",
  "complexity": "medium",
  "priority": "medium",
  "estimated_hours": 40,
  "key_requirements": "Cloud developer with Azure experience"
}}

Query: ".NET Core developer"
{{
  "required_skills": [".NET Core"],
  "all_skills_required": true,
  "related_skills": ["C#", "ASP.NET", "Azure", "SQL Server"],
  "department": "Engineering",
  "complexity": "medium",
  "priority": "medium",
  "estimated_hours": 40,
  "key_requirements": "Backend developer with .NET Core"
}}

Query: "Find me a React or Python developer"
{{
  "required_skills": ["React", "Python"],
  "all_skills_required": false,
  "related_skills": ["JavaScript", "TypeScript"],
  "department": "Engineering",
  "complexity": "medium",
  "priority": "medium",
  "estimated_hours": 40,
  "key_requirements": "Developer with React or Python skills"
}}

Return ONLY valid JSON, no other text.
"""
        
        try:
            # Try Gemini first (free and fast)
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Clean up response (remove markdown code blocks if present)
                if result_text.startswith('```json'):
                    result_text = result_text.split('```json')[1].split('```')[0].strip()
                elif result_text.startswith('```'):
                    result_text = result_text.split('```')[1].split('```')[0].strip()
                
                result = json.loads(result_text)
                return result
            
            # Fallback to OpenAI
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a task analysis assistant. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result_text = response.choices[0].message.content.strip()
                
                # Clean up response
                if result_text.startswith('```json'):
                    result_text = result_text.split('```json')[1].split('```')[0].strip()
                elif result_text.startswith('```'):
                    result_text = result_text.split('```')[1].split('```')[0].strip()
                
                result = json.loads(result_text)
                return result
            
            else:
                # No AI available, use simple keyword extraction
                return self._fallback_analysis(task_description, task_title)
        
        except Exception as e:
            print(f"❌ AI analysis error: {e}")
            print("💡 Falling back to keyword-based analysis")
            return self._fallback_analysis(task_description, task_title)
    
    def _fallback_analysis(self, description: str, title: str) -> Dict:
        """Fallback keyword-based analysis when AI is not available"""
        text = f"{title} {description}".lower()
        
        # Skill detection
        skill_keywords = {
            'react': ['react', 'reactjs'],
            'python': ['python'],
            'javascript': ['javascript', 'js'],
            'typescript': ['typescript', 'ts'],
            'node.js': ['node', 'nodejs'],
            'aws': ['aws', 'amazon web services'],
            'docker': ['docker', 'container'],
            'kubernetes': ['kubernetes', 'k8s'],
            'api': ['api', 'rest', 'graphql'],
        }
        
        detected_skills = []
        for skill, keywords in skill_keywords.items():
            if any(kw in text for kw in keywords):
                detected_skills.append(skill.title())
        
        # Department detection
        if any(word in text for word in ['frontend', 'ui', 'ux', 'react', 'vue']):
            department = 'Engineering'
        elif any(word in text for word in ['backend', 'api', 'database']):
            department = 'Engineering'
        elif any(word in text for word in ['devops', 'infrastructure', 'deploy']):
            department = 'DevOps'
        elif any(word in text for word in ['data', 'analytics', 'ml', 'ai']):
            department = 'Data'
        else:
            department = 'Engineering'
        
        # Complexity detection
        if any(word in text for word in ['complex', 'architecture', 'system', 'enterprise']):
            complexity = 'high'
        elif any(word in text for word in ['simple', 'basic', 'small']):
            complexity = 'low'
        else:
            complexity = 'medium'
        
        # Priority detection
        if any(word in text for word in ['urgent', 'critical', 'asap', 'immediately']):
            priority = 'critical'
        elif any(word in text for word in ['important', 'priority', 'soon']):
            priority = 'high'
        else:
            priority = 'medium'
        
        return {
            'required_skills': detected_skills or ['General Development'],
            'department': department,
            'complexity': complexity,
            'priority': priority,
            'estimated_hours': 40,
            'key_requirements': title or 'Task analysis'
        }
    
    def match_resources_to_task(self, task_analysis: Dict, resources: List[Dict]) -> List[Dict]:
        """
        Intelligently match resources to a task based on AI analysis
        Respects "all_skills_required" flag for AND logic
        Returns resources ranked by suitability
        """
        required_skills = [s.lower() for s in task_analysis.get('required_skills', [])]
        all_skills_required = task_analysis.get('all_skills_required', False)
        related_skills = [s.lower() for s in task_analysis.get('related_skills', [])]
        complexity = task_analysis.get('complexity', 'medium')
        
        # Skill relationship mapping for fuzzy matching - EXPANDED
        skill_aliases = {
            '.net': ['asp.net', 'c#', 'dotnet', 'aspnet', 'c# .net', '.net core', 'dotnet core', 'net core', 'net'],
            'asp.net': ['.net', 'c#', 'dotnet', 'aspnet', '.net core'],
            '.net core': ['.net', 'asp.net', 'dotnet', 'dotnet core', 'net core'],
            'azure': ['microsoft azure', 'ms azure', 'azure cloud', 'windows azure'],
            'aws': ['amazon web services', 'amazon aws', 'aws cloud'],
            'gcp': ['google cloud', 'google cloud platform', 'gcloud'],
            'node.js': ['nodejs', 'node', 'node js'],
            'react': ['reactjs', 'react.js', 'react js'],
            'vue': ['vuejs', 'vue.js', 'vue js'],
            'angular': ['angularjs', 'angular.js', 'angular js'],
            'python': ['python3', 'python 3', 'py'],
            'javascript': ['js', 'ecmascript', 'es6', 'es2015'],
            'typescript': ['ts'],
            'kubernetes': ['k8s', 'k8'],
            'docker': ['containers', 'containerization'],
            'c#': ['csharp', 'c sharp', 'c-sharp'],
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'nlp': ['natural language processing', 'language processing'],
            'chatbot': ['chat bot', 'conversational ai', 'bot'],
        }
        
        def normalize_skill(skill: str) -> set:
            """Return a set of all possible names for a skill"""
            skill_lower = skill.lower()
            aliases = {skill_lower}
            
            # Check if this skill has aliases
            for main_skill, alias_list in skill_aliases.items():
                if skill_lower == main_skill or skill_lower in alias_list:
                    aliases.add(main_skill)
                    aliases.update(alias_list)
            
            return aliases
        
        scored_resources = []
        
        for resource in resources:
            score = 0
            reasons = []
            
            # Normalize resource skills
            resource_skills_normalized = set()
            for skill in resource.get('skills', []):
                resource_skills_normalized.update(normalize_skill(skill))
            
            # Skill matching with AND/OR logic
            if required_skills:
                matched_skills = []
                missing_skills = []
                
                for required_skill in required_skills:
                    required_skill_normalized = normalize_skill(required_skill)
                    
                    if resource_skills_normalized & required_skill_normalized:
                        matched_skills.append(required_skill)
                    else:
                        missing_skills.append(required_skill)
                
                # Calculate skill match ratio
                skill_match_ratio = len(matched_skills) / len(required_skills) if required_skills else 0
                
                # Apply AND logic: if all_skills_required=true, must have ALL skills
                if all_skills_required:
                    if missing_skills:
                        # Missing required skills - low score or skip
                        continue  # Skip this resource entirely
                    else:
                        # Has ALL required skills - high score!
                        score += 60  # Base score for having all skills
                        reasons.append(f"✅ Has ALL {len(matched_skills)} required skills")
                else:
                    # OR logic: any matching skill is good
                    score += skill_match_ratio * 50
                    if matched_skills:
                        reasons.append(f"Has {len(matched_skills)}/{len(required_skills)} required skills")
                
                # Bonus for matching related skills
                related_matched = 0
                for related_skill in related_skills:
                    related_normalized = normalize_skill(related_skill)
                    if resource_skills_normalized & related_normalized:
                        related_matched += 1
                
                if related_matched > 0:
                    score += related_matched * 5
                    reasons.append(f"Has {related_matched} related skills")
            
            # Availability check
            if resource.get('availability') == 'available':
                score += 20
                reasons.append("Currently available")
            else:
                score += 5  # Still include busy resources but with lower score
            
            # Workload consideration
            workload = resource.get('current_workload', 50)
            if workload < 50:
                score += 15
                reasons.append("Low workload")
            elif workload < 70:
                score += 10
                reasons.append("Moderate workload")
            else:
                score += 5
            
            # Expertise level matching
            expertise = resource.get('expertise_level', 'mid')
            if complexity == 'high' and expertise in ['senior', 'expert']:
                score += 15
                reasons.append("Senior level for complex task")
            elif complexity == 'low' and expertise in ['junior', 'mid']:
                score += 10
                reasons.append("Appropriate experience level")
            elif complexity == 'medium' and expertise in ['mid', 'senior']:
                score += 10
                reasons.append("Good experience match")
            
            # Department match
            task_dept = task_analysis.get('department', '')
            if resource.get('department', '') == task_dept:
                score += 5
                reasons.append("Same department")
            
            # Only include resources with meaningful scores
            if score > 20:  # Minimum threshold
                resource_copy = resource.copy()
                resource_copy['match_score'] = min(score / 100, 1.0)  # Normalize to 0-1
                resource_copy['recommendation_reason'] = ' • '.join(reasons)
                scored_resources.append(resource_copy)
        
        # Sort by score descending
        scored_resources.sort(key=lambda x: x['match_score'], reverse=True)
        
        return scored_resources
    
    def generate_task_summary(self, task_analysis: Dict) -> str:
        """Generate a human-readable summary of task analysis"""
        skills = ', '.join(task_analysis.get('required_skills', []))
        complexity = task_analysis.get('complexity', 'medium')
        priority = task_analysis.get('priority', 'medium')
        hours = task_analysis.get('estimated_hours', 'unknown')
        
        return f"**Skills needed:** {skills} | **Complexity:** {complexity} | **Priority:** {priority} | **Est. hours:** {hours}"


# Singleton instance
ai_service = AIService()

