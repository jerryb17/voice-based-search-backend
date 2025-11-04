from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from nlp_processor import NLPProcessor
from ai_service import ai_service

app = Flask(__name__)
CORS(app)

# Initialize NLP processor and AI service
nlp_processor = NLPProcessor()
print("Initializing AI service for intelligent task analysis...")

def load_resources():
    """Load resources from JSON file"""
    try:
        with open('resources_data.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading resources: {e}")
        return []

def load_tasks():
    """Load tasks from JSON file"""
    try:
        with open('tasks_data.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tasks: {e}")
        return []

def save_resources():
    """Save resources to JSON file"""
    try:
        with open('resources_data.json', 'w') as f:
            json.dump(resources, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving resources: {e}")
        return False

def save_tasks():
    """Save tasks to JSON file"""
    try:
        with open('tasks_data.json', 'w') as f:
            json.dump(tasks, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving tasks: {e}")
        return False

# Load data
resources = load_resources()
tasks = load_tasks()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Backend is running'}), 200

@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get all resources with optional filtering"""
    try:
        availability = request.args.get('availability')
        skill = request.args.get('skill')
        expertise_level = request.args.get('expertise_level')
        department = request.args.get('department')
        
        filtered_resources = resources.copy()
        
        if availability:
            filtered_resources = [r for r in filtered_resources if r['availability'] == availability]
        
        if skill:
            filtered_resources = [
                r for r in filtered_resources 
                if skill.lower() in [s.lower() for s in r['skills']]
            ]
        
        if expertise_level:
            filtered_resources = [
                r for r in filtered_resources 
                if r['expertise_level'] == expertise_level
            ]
        
        if department:
            filtered_resources = [
                r for r in filtered_resources 
                if r['department'] == department
            ]
        
        return jsonify({
            'success': True,
            'count': len(filtered_resources),
            'resources': filtered_resources
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/resources/<int:resource_id>', methods=['GET'])
def get_resource(resource_id):
    """Get a specific resource by ID"""
    try:
        resource = next((r for r in resources if r['id'] == resource_id), None)
        
        if resource:
            return jsonify({'success': True, 'resource': resource}), 200
        else:
            return jsonify({'success': False, 'error': 'Resource not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks with optional filtering"""
    try:
        status = request.args.get('status')
        priority = request.args.get('priority')
        
        filtered_tasks = tasks.copy()
        
        if status:
            filtered_tasks = [t for t in filtered_tasks if t['status'] == status]
        
        if priority:
            filtered_tasks = [t for t in filtered_tasks if t['priority'] == priority]
        
        return jsonify({
            'success': True,
            'count': len(filtered_tasks),
            'tasks': filtered_tasks
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tasks/<int:task_id>/assign', methods=['POST'])
def assign_task(task_id):
    """Assign a resource to a task and update workload"""
    try:
        data = request.get_json()
        resource_id = data.get('resource_id')
        
        # Find task
        task = next((t for t in tasks if t['id'] == task_id), None)
        if not task:
            return jsonify({'success': False, 'error': 'Task not found'}), 404
        
        # Find resource
        resource = next((r for r in resources if r['id'] == resource_id), None)
        if not resource:
            return jsonify({'success': False, 'error': 'Resource not found'}), 404
        
        # Calculate workload increase based on estimated hours
        # Assuming 40 hours = 100% capacity per week
        hours_per_week = 40
        workload_increase = (task.get('estimated_hours', 40) / hours_per_week) * 100
        
        # Update resource workload
        new_workload = min(resource['current_workload'] + workload_increase, 100)
        resource['current_workload'] = round(new_workload, 1)
        
        # Update availability based on workload
        if resource['current_workload'] >= 80:
            resource['availability'] = 'busy'
        else:
            resource['availability'] = 'available'
        
        # Add task to resource's assigned tasks
        if 'assigned_tasks' not in resource:
            resource['assigned_tasks'] = []
        resource['assigned_tasks'].append(task_id)
        
        # Update task
        task['assigned_resource'] = resource_id
        task['status'] = 'assigned'
        
        # Save to database (JSON file)
        save_resources()
        save_tasks()
        
        return jsonify({
            'success': True,
            'task': task,
            'resource': resource,
            'message': f'Task assigned to {resource["name"]}. Workload updated to {resource["current_workload"]}%'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tasks/<int:task_id>/unassign', methods=['POST'])
def unassign_task(task_id):
    """Unassign a task from a resource and update workload"""
    try:
        task = next((t for t in tasks if t['id'] == task_id), None)
        if not task:
            return jsonify({'success': False, 'error': 'Task not found'}), 404
        
        if not task.get('assigned_resource'):
            return jsonify({'success': False, 'error': 'Task is not assigned'}), 400
        
        # Find resource
        resource = next((r for r in resources if r['id'] == task['assigned_resource']), None)
        if resource:
            # Calculate workload decrease
            hours_per_week = 40
            workload_decrease = (task.get('estimated_hours', 40) / hours_per_week) * 100
            
            # Update resource workload
            new_workload = max(resource['current_workload'] - workload_decrease, 0)
            resource['current_workload'] = round(new_workload, 1)
            
            # Update availability
            if resource['current_workload'] < 80:
                resource['availability'] = 'available'
            
            # Remove task from assigned tasks
            if 'assigned_tasks' in resource and task_id in resource['assigned_tasks']:
                resource['assigned_tasks'].remove(task_id)
        
        # Update task
        task['assigned_resource'] = None
        task['status'] = 'pending'
        
        # Save to database
        save_resources()
        save_tasks()
        
        return jsonify({
            'success': True,
            'task': task,
            'resource': resource
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_resources():
    """Search resources using NLP"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        results = nlp_processor.search_candidates(query, top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'resources': results
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_resources():
    """Recommend resources for a specific task using AI analysis"""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        task_description = data.get('task_description', '')
        task_title = data.get('task_title', '')
        top_k = data.get('top_k', 5)
        use_ai = data.get('use_ai', True)
        
        if task_id:
            task = next((t for t in tasks if t['id'] == task_id), None)
            if task:
                task_title = task['title']
                task_description = task['description']
        
        if not task_description:
            return jsonify({'success': False, 'error': 'Task description is required'}), 400
        
        # Use AI to analyze the task
        if use_ai:
            print(f"🤖 Using AI to analyze task: {task_title}")
            task_analysis = ai_service.analyze_task(task_description, task_title)
            print(f"✅ AI Analysis: {task_analysis}")
            
            # Use AI matching
            recommendations = ai_service.match_resources_to_task(task_analysis, resources)
            recommendations = recommendations[:top_k]
            
            # Add AI analysis to response
            analysis_summary = ai_service.generate_task_summary(task_analysis)
            
            return jsonify({
                'success': True,
                'task': task_description,
                'task_analysis': task_analysis,
                'analysis_summary': analysis_summary,
                'count': len(recommendations),
                'recommendations': recommendations,
                'ai_powered': True
            }), 200
        else:
            # Fallback to NLP-only matching
            recommendations = nlp_processor.recommend_for_task(task_description, top_k)
            
            return jsonify({
                'success': True,
                'task': task_description,
                'count': len(recommendations),
                'recommendations': recommendations,
                'ai_powered': False
            }), 200
        
    except Exception as e:
        print(f"Error in recommend_resources: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about resources and tasks"""
    try:
        total_resources = len(resources)
        available_resources = len([r for r in resources if r['availability'] == 'available'])
        busy_resources = total_resources - available_resources
        
        total_tasks = len(tasks)
        pending_tasks = len([t for t in tasks if t['status'] == 'pending'])
        assigned_tasks = len([t for t in tasks if t['status'] == 'assigned'])
        
        # Average workload
        avg_workload = sum(r['current_workload'] for r in resources) / total_resources if total_resources > 0 else 0
        
        # Skills distribution
        skill_counts = {}
        for resource in resources:
            for skill in resource['skills']:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'success': True,
            'stats': {
                'total_resources': total_resources,
                'available_resources': available_resources,
                'busy_resources': busy_resources,
                'total_tasks': total_tasks,
                'pending_tasks': pending_tasks,
                'assigned_tasks': assigned_tasks,
                'average_workload': round(avg_workload, 1),
                'top_skills': [{'skill': skill, 'count': count} for skill, count in top_skills]
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-task', methods=['POST'])
def analyze_task():
    """Analyze a task description using AI"""
    try:
        data = request.get_json()
        task_description = data.get('task_description', '')
        task_title = data.get('task_title', '')
        
        if not task_description:
            return jsonify({'success': False, 'error': 'Task description is required'}), 400
        
        analysis = ai_service.analyze_task(task_description, task_title)
        summary = ai_service.generate_task_summary(analysis)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'summary': summary
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Resource Allocation System Backend")
    print("="*60)
    print(f"📊 Loaded {len(resources)} resources")
    print(f"📋 Loaded {len(tasks)} tasks")
    print("🧠 Initializing NLP processor...")
    print("🤖 AI Service Status:")
    print(f"   - Gemini: {'✅ Ready' if ai_service.gemini_model else '❌ Not configured'}")
    print(f"   - OpenAI: {'✅ Ready' if ai_service.openai_client else '❌ Not configured'}")
    print("\n📡 Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/resources - Get all resources")
    print("  GET  /api/resources/<id> - Get specific resource")
    print("  GET  /api/tasks - Get all tasks")
    print("  POST /api/tasks/<id>/assign - Assign resource to task")
    print("  POST /api/search - Search resources using NLP")
    print("  POST /api/recommend - Get AI-powered recommendations 🤖")
    print("  POST /api/analyze-task - Analyze task with AI 🤖")
    print("  GET  /api/stats - Get system statistics")
    print("\n" + "="*60)
    print("✅ Server running on http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
