"""
Flask API for Body Composition Analysis
REST API endpoints for analyzing body composition from images
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import uuid

# Import custom modules
try:
    from body_composition_analyzer import get_body_analyzer
    from database import get_database
    ANALYZER_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import body composition analyzer: {e}")
    ANALYZER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads/body_composition'
app.config['PROCESSED_FOLDER'] = 'processed_images'

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'analyzer_available': ANALYZER_AVAILABLE,
        'version': '1.0.0'
    })

@app.route('/analyze-body-composition', methods=['POST'])
def analyze_body_composition():
    """Analyze body composition from uploaded image(s)."""
    if not ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'Body composition analyzer not available',
            'success': False
        }), 503
    
    try:
        # Check if user_id is provided
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required', 'success': False}), 400
        
        # Check if file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided', 'success': False}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS),
                'success': False
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process additional images if provided
        additional_images = {}
        for image_type in ['front_image', 'side_image']:
            if image_type in request.files:
                additional_file = request.files[image_type]
                if additional_file.filename != '' and allowed_file(additional_file.filename):
                    additional_filename = secure_filename(additional_file.filename)
                    additional_unique_filename = f"{timestamp}_{image_type}_{uuid.uuid4().hex[:8]}_{additional_filename}"
                    additional_file_path = os.path.join(app.config['UPLOAD_FOLDER'], additional_unique_filename)
                    additional_file.save(additional_file_path)
                    additional_images[image_type.replace('_image', '')] = additional_file_path
        
        # Analyze body composition
        analyzer = get_body_analyzer()
        analysis_result = analyzer.analyze_image(
            image_path=file_path,
            user_id=user_id,
            additional_images=additional_images if additional_images else None
        )
        
        if analysis_result.get('success', False):
            return jsonify({
                'success': True,
                'analysis_id': analysis_result['analysis_id'],
                'body_composition': {
                    'body_fat_percentage': analysis_result['body_fat_percentage'],
                    'muscle_mass_percentage': analysis_result['muscle_mass_percentage'],
                    'visceral_fat_level': analysis_result['visceral_fat_level'],
                    'bmr_estimated': analysis_result['bmr_estimated'],
                    'body_shape': analysis_result['body_shape'],
                    'confidence': analysis_result['confidence']
                },
                'measurements': analysis_result.get('measurements', {}),
                'breakdown': analysis_result.get('breakdown', {}),
                'processed_image_path': analysis_result.get('processed_image_path', ''),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': analysis_result.get('error', 'Analysis failed'),
                'timestamp': datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        logger.error(f"Error in body composition analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/user/<user_id>/body-composition-history', methods=['GET'])
def get_body_composition_history(user_id):
    """Get body composition analysis history for a user."""
    if not ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'Body composition analyzer not available',
            'success': False
        }), 503
    
    try:
        days = request.args.get('days', 90, type=int)
        
        db = get_database()
        history = db.get_body_composition_history(user_id, days)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'history': history,
            'count': len(history),
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting body composition history: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/user/<user_id>/latest-body-composition', methods=['GET'])
def get_latest_body_composition(user_id):
    """Get latest body composition analysis for a user."""
    if not ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'Body composition analyzer not available',
            'success': False
        }), 503
    
    try:
        db = get_database()
        latest = db.get_latest_body_composition(user_id)
        
        if latest:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'analysis': latest,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'analysis': None,
                'message': 'No body composition analysis found for user',
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error getting latest body composition: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/user/<user_id>/composition-progress', methods=['GET'])
def get_composition_progress(user_id):
    """Get body composition progress over time."""
    if not ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'Body composition analyzer not available',
            'success': False
        }), 503
    
    try:
        period_days = request.args.get('period_days', 30, type=int)
        
        db = get_database()
        progress = db.calculate_composition_progress(user_id, period_days)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'progress': progress,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error calculating composition progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/compare-analyses', methods=['POST'])
def compare_analyses():
    """Compare two body composition analyses."""
    if not ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'Body composition analyzer not available',
            'success': False
        }), 503
    
    try:
        data = request.get_json()
        
        required_fields = ['user_id', 'analysis_id1', 'analysis_id2']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        analyzer = get_body_analyzer()
        comparison = analyzer.compare_analyses(
            user_id=data['user_id'],
            analysis_id1=data['analysis_id1'],
            analysis_id2=data['analysis_id2']
        )
        
        if 'error' in comparison:
            return jsonify({
                'success': False,
                'error': comparison['error'],
                'timestamp': datetime.now().isoformat()
            }), 400
        
        return jsonify({
            'success': True,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error comparing analyses: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/processed-image/<path:filename>', methods=['GET'])
def get_processed_image(filename):
    """Serve processed analysis images."""
    try:
        file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving processed image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-limits', methods=['GET'])
def get_upload_limits():
    """Get upload limitations and supported formats."""
    return jsonify({
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'multiple_images_supported': True,
        'image_types': {
            'main': 'Primary body image for analysis',
            'front': 'Front view image (optional)',
            'side': 'Side view image (optional)'
        }
    })

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'max_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle not found error."""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    # Development server
    logger.info("Starting Body Composition Analysis API...")
    logger.info(f"Analyzer available: {ANALYZER_AVAILABLE}")
    
    app.run(
        host='0.0.0.0',
        port=5001,  # Different port to avoid conflicts
        debug=True
    )
