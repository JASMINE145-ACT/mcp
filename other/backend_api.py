"""
Flask API Backend for Data Analysis Agent
Provides REST API endpoints to interact with agent.py
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
import sys
import json
from werkzeug.utils import secure_filename
import glob
from datetime import datetime

# Add parent directory to path to import agent and function modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import DataAIAgent
from function import profile_dataframe_simple

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configuration
# Use absolute path for upload folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state (In production, use Redis or database)
sessions = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Backend is running"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload CSV file and create analysis session"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Create profile
            profile_data = profile_dataframe_simple(df)
            
            # Create agent
            agent = DataAIAgent(df)
            
            # Create session
            session_id = timestamp
            sessions[session_id] = {
                'df': df,
                'agent': agent,
                'profile_data': profile_data,
                'filepath': filepath,
                'filename': filename,
                'conversation_state': None,
                'analysis_history': []
            }
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "filename": filename,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "profile": profile_data
            })
        else:
            return jsonify({"error": "Invalid file type. Only CSV allowed"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session information"""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    return jsonify({
        "session_id": session_id,
        "filename": session['filename'],
        "shape": session['df'].shape,
        "columns": session['df'].columns.tolist(),
        "analysis_count": len(session['analysis_history'])
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze data based on user question"""
    try:
        data = request.json
        session_id = data.get('session_id')
        question = data.get('question')
        user_feedback = data.get('feedback', None)
        
        if not session_id or not question:
            return jsonify({"error": "Missing session_id or question"}), 400
        
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        
        session = sessions[session_id]
        agent = session['agent']
        profile_data = session['profile_data']
        
        # Call agent analysis
        if user_feedback:
            # Continue existing conversation
            result = agent.langgraph_analysis(
                question=question,
                dataset_info=profile_data,
                user_feedback=user_feedback,
                existing_state=session['conversation_state']
            )
        else:
            # New analysis
            result = agent.langgraph_analysis(question, profile_data)
        
        # Update conversation state
        conv_state = result.get('conversation_state', {})
        session['conversation_state'] = {
            'question': question,
            'dataset_info': profile_data,
            'plan': result.get('plan', ''),
            'code': result.get('code', ''),
            'execution_result': result.get('execution_result', ''),
            'validation': result.get('validation', ''),
            'final_result': result.get('final_result', {}),
            'conversation_history': conv_state.get('conversation_history', []),
            'plan_iterations': conv_state.get('plan_iterations', []),
            'user_feedback': '',
            'plan_confirmed': conv_state.get('plan_confirmed', False)
        }
        
        # Check if needs user input
        needs_confirmation = conv_state.get('needs_user_input', False)
        
        response_data = {
            "success": True,
            "needs_confirmation": needs_confirmation,
            "question": question,
            "plan": result.get('plan', ''),
            "code": result.get('code', ''),
            "execution_result": result.get('execution_result', ''),
            "validation": result.get('validation', ''),
            "conversation_state": conv_state
        }
        
        # If analysis is complete, save to history
        if not needs_confirmation and result.get('validation'):
            session['analysis_history'].append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'result': result
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/confirm', methods=['POST'])
def confirm_plan():
    """Confirm plan and execute analysis"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        
        session = sessions[session_id]
        agent = session['agent']
        profile_data = session['profile_data']
        conversation_state = session['conversation_state']
        
        # Mark as confirmed
        conversation_state['plan_confirmed'] = True
        conversation_state['user_feedback'] = "用户确认计划，请执行"
        
        # Execute analysis
        result = agent.langgraph_analysis(
            question=conversation_state['question'],
            dataset_info=profile_data,
            user_feedback=conversation_state['user_feedback'],
            existing_state=conversation_state
        )
        
        # Save to history
        session['analysis_history'].append({
            'timestamp': datetime.now().isoformat(),
            'question': conversation_state['question'],
            'result': result
        })
        
        # Reset conversation state
        session['conversation_state'] = None
        
        return jsonify({
            "success": True,
            "question": conversation_state['question'],
            "code": result.get('code', ''),
            "execution_result": result.get('execution_result', ''),
            "validation": result.get('validation', '')
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get analysis history for a session"""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    return jsonify({
        "success": True,
        "history": session['analysis_history']
    })

@app.route('/api/plots', methods=['GET'])
def list_plots():
    """List available plot files"""
    png_files = glob.glob("*.png")
    png_files.sort(key=os.path.getmtime, reverse=True)
    
    return jsonify({
        "success": True,
        "plots": png_files
    })

@app.route('/api/plot/<filename>', methods=['GET'])
def get_plot(filename):
    """Get a specific plot file"""
    try:
        return send_file(filename, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    session_list = []
    for session_id, session in sessions.items():
        session_list.append({
            'session_id': session_id,
            'filename': session['filename'],
            'shape': session['df'].shape,
            'analysis_count': len(session['analysis_history'])
        })
    
    return jsonify({
        "success": True,
        "sessions": session_list
    })

if __name__ == '__main__':
    print("🚀 Starting Flask Backend API...")
    print("📊 Endpoint: http://localhost:8000")
    print("📝 API Docs: http://localhost:8000/api/health")
    app.run(host='0.0.0.0', port=8000, debug=True)

