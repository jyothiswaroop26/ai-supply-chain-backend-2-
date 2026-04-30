"""
API routes definition
"""
from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)

@api_bp.route('/status', methods=['GET'])
def status():
    """API status endpoint"""
    return jsonify({'message': 'API is running'}), 200
