import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Import the updated Grad-CAM function
from grad_cam_utils import generate_gradcam_with_prediction

# MongoDB logging (optional - will work without MongoDB)
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    st.sidebar.info("📝 MongoDB not available - predictions will be stored locally only")

# Email functionality with environment variables
try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    EMAIL_AVAILABLE = True
    SMTP_USER = os.getenv('SMTP_USER')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    
    if not SMTP_USER or not SMTP_PASSWORD:
        EMAIL_AVAILABLE = False
        st.sidebar.warning("📧 Email service: Not configured")
    else:
        st.sidebar.success("📧 Email service: Configured")
        
except ImportError:
    EMAIL_AVAILABLE = False
    st.sidebar.error("📧 Email packages not available")

# Configure Streamlit page
st.set_page_config(
    page_title="Pneumonia Detection Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Tailwind-like styling with medical background
def load_css():
    st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Professional Medical Background */
    .stApp {
        background-image: 
            linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.9) 25%, 
                rgba(51, 65, 85, 0.85) 50%, 
                rgba(71, 85, 105, 0.9) 75%, 
                rgba(15, 23, 42, 0.95) 100%
            ),
            radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
            url('https://images.unsplash.com/photo-1631815588090-d4bfec5b1ccb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover, 1000px 1000px, 800px 800px, cover;
        background-position: center, 25% 25%, 75% 75%, center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Alternative Professional Medical Backgrounds */
    /*
    Option 1: Modern Hospital Interior
    .stApp {
        background-image: 
            linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.88) 50%, rgba(51, 65, 85, 0.92) 100%),
            url('https://images.unsplash.com/photo-1586773860418-d37222d8fce3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
    }
    
    Option 2: Medical Technology & AI
    .stApp {
        background-image: 
            linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.85) 100%),
            url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
    }
    
    Option 3: Clean Medical Environment
    .stApp {
        background-image: 
            linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.82) 100%),
            url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
    }
    
    Option 4: Abstract Medical Pattern (No external image)
    .stApp {
        background: 
            linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #475569 75%, #0f172a 100%),
            repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(59,130,246,0.05) 2px, rgba(59,130,246,0.05) 4px),
            radial-gradient(circle at 20% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 50%);
    }
    */
    
    /* Enhanced main header with medical X-ray theme */
    .main-header {
        background: 
            linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.92) 25%, 
                rgba(51, 65, 85, 0.88) 50%, 
                rgba(71, 85, 105, 0.92) 75%, 
                rgba(15, 23, 42, 0.95) 100%
            ),
            radial-gradient(ellipse at top left, rgba(59, 130, 246, 0.2) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(16, 185, 129, 0.2) 0%, transparent 50%);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 2px solid rgba(148, 163, 184, 0.3);
        padding: 3rem 2rem;
        border-radius: 2rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 25px 50px -12px rgba(15, 23, 42, 0.6),
            0 0 0 1px rgba(148, 163, 184, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 50px rgba(59, 130, 246, 0.15);
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: medical-glow 4s ease-in-out infinite;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, transparent 30%, rgba(147, 197, 253, 0.1) 50%, transparent 70%, rgba(255, 255, 255, 0.1) 100%),
            radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.1) 2px, transparent 2px),
            radial-gradient(circle at 80% 80%, rgba(147, 197, 253, 0.2) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 60px 60px;
        pointer-events: none;
        opacity: 0.6;
    }
    
    /* Enhanced prediction cards with glassmorphism */
    .prediction-card {
        background: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(148, 163, 184, 0.3);
        padding: 2.5rem;
        border-radius: 1.5rem;
        box-shadow: 
            0 20px 25px -5px rgba(15, 23, 42, 0.15),
            0 10px 10px -5px rgba(15, 23, 42, 0.08),
            0 0 0 1px rgba(148, 163, 184, 0.2);
        margin: 1.5rem 0;
        border-left: 6px solid #0f172a;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.15),
            0 0 0 1px rgba(59, 130, 246, 0.2);
    }
    
    .normal-card {
        border-left-color: #10b981;
        background: 
            linear-gradient(135deg, rgba(236, 253, 245, 0.95) 0%, rgba(240, 253, 244, 0.98) 100%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.1) 0%, transparent 50%);
        box-shadow: 
            0 20px 25px -5px rgba(16, 185, 129, 0.15),
            0 10px 10px -5px rgba(16, 185, 129, 0.08),
            0 0 0 1px rgba(16, 185, 129, 0.2),
            0 0 20px rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .normal-card:hover {
        box-shadow: 
            0 25px 50px -12px rgba(16, 185, 129, 0.25),
            0 0 0 1px rgba(16, 185, 129, 0.4),
            0 0 30px rgba(16, 185, 129, 0.2);
        transform: translateY(-3px) scale(1.02);
    }
    
    .pneumonia-card {
        border-left-color: #ef4444;
        background: 
            linear-gradient(135deg, rgba(254, 242, 242, 0.95) 0%, rgba(254, 247, 247, 0.98) 100%),
            radial-gradient(circle at top right, rgba(239, 68, 68, 0.1) 0%, transparent 50%);
        box-shadow: 
            0 20px 25px -5px rgba(239, 68, 68, 0.15),
            0 10px 10px -5px rgba(239, 68, 68, 0.08),
            0 0 0 1px rgba(239, 68, 68, 0.2),
            0 0 20px rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .pneumonia-card:hover {
        box-shadow: 
            0 25px 50px -12px rgba(239, 68, 68, 0.25),
            0 0 0 1px rgba(239, 68, 68, 0.4),
            0 0 30px rgba(239, 68, 68, 0.2);
        transform: translateY(-3px) scale(1.02);
    }
    
    /* Enhanced upload section */
    .upload-section {
        background: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 3px dashed rgba(203, 213, 225, 0.8);
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(59, 130, 246, 0.02) 0%, transparent 50%, rgba(59, 130, 246, 0.02) 100%);
        pointer-events: none;
    }
    
    .upload-section:hover {
        border-color: rgba(59, 130, 246, 0.8);
        background: rgba(241, 245, 249, 0.98);
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.15);
    }
    
    /* Enhanced accuracy badge */
    .accuracy-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        margin: 0.75rem;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced confidence indicators */
    .confidence-high { 
        color: #059669; 
        font-weight: 700; 
        text-shadow: 0 1px 2px rgba(5, 150, 105, 0.2);
    }
    .confidence-medium { 
        color: #d97706; 
        font-weight: 700; 
        text-shadow: 0 1px 2px rgba(217, 119, 6, 0.2);
    }
    .confidence-low { 
        color: #dc2626; 
        font-weight: 700; 
        text-shadow: 0 1px 2px rgba(220, 38, 38, 0.2);
    }
    
    /* Enhanced Streamlit components */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    
    /* Enhanced file uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(59, 130, 246, 0.6) !important;
        background: rgba(255, 255, 255, 0.98) !important;
    }
    
    /* File uploader text visibility */
    .stFileUploader label, .stFileUploader div, .stFileUploader span {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        text-shadow: none !important;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(203, 213, 225, 0.3);
    }
    
    /* Enhanced metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 0.75rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Enhanced expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 0.5rem;
        border: 1px solid rgba(203, 213, 225, 0.3);
    }
    
    /* Enhanced dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Professional Typography for Medical Application */
    
    /* Main Headers - Professional Medical Style */
    .stMarkdown h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6) !important;
        line-height: 1.3 !important;
        margin-bottom: 1rem !important;
    }
    
    .stMarkdown h2 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) !important;
        line-height: 1.4 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .stMarkdown h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4) !important;
        line-height: 1.4 !important;
        margin-bottom: 0.6rem !important;
    }
    
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
        line-height: 1.5 !important;
    }
    
    /* Body Text - Professional Readability */
    .stMarkdown p {
        color: #cbd5e1 !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Streamlit Components - Professional Styling */
    .stButton > button {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: none !important;
        letter-spacing: 0.025em !important;
    }
    
    /* Labels and Form Elements */
    .stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Metrics - Clean Professional Display */
    .stMetric {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="metric-label"] {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    /* Alert Messages - Professional Medical Styling */
    .stSuccess {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #065f46 !important;
    }
    
    .stWarning {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #92400e !important;
    }
    
    .stError {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #991b1b !important;
    }
    
    .stInfo {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #1e40af !important;
    }
    
    /* Sidebar - Professional Medical Theme */
    .css-1d391kg .stMarkdown {
        color: #1e293b !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .css-1d391kg h3 {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* White Card Content - Dark Text for Readability */
    .prediction-card * {
        color: #1e293b !important;
        font-weight: 500 !important;
        text-shadow: none !important;
    }
    
    .prediction-card h2 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    .prediction-card h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    .prediction-card p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Specific styling for placeholder content */
    .prediction-card div[style*="background: rgba(248, 250, 252"] * {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    .upload-section * {
        color: #1e293b !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }
    
    .upload-section h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    .upload-section p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* File Uploader - Professional Styling */
    .stFileUploader * {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Expander Headers - Professional */
    .streamlit-expanderHeader {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Dataframe - Clean Professional */
    .stDataFrame * {
        color: #1e293b !important;
        font-weight: 400 !important;
        font-size: 0.9rem !important;
    }
    
    /* Caption Text - Subtle Professional */
    .stCaption {
        color: #94a3b8 !important;
        font-weight: 400 !important;
        font-size: 0.85rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Medical icons and animations */
    @keyframes pulse-medical {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    @keyframes medical-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.6); }
    }
    
    @keyframes xray-scan {
        0% { background-position: -100% 0; }
        100% { background-position: 100% 0; }
    }
    
    .medical-icon {
        animation: pulse-medical 3s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.5));
    }
    
    /* X-ray scan effect for cards */
    .prediction-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(59, 130, 246, 0.1) 20%, 
            rgba(147, 197, 253, 0.2) 50%, 
            rgba(59, 130, 246, 0.1) 80%, 
            transparent
        );
        animation: xray-scan 3s ease-in-out infinite;
        pointer-events: none;
    }
    
    /* Medical grid pattern overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.1) 2px, transparent 2px),
            radial-gradient(circle at 75% 75%, rgba(147, 197, 253, 0.1) 1px, transparent 1px);
        background-size: 50px 50px, 30px 30px;
        pointer-events: none;
        z-index: -1;
        opacity: 0.3;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
        
        .upload-section {
            padding: 2rem 1rem;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(248, 250, 252, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(59, 130, 246, 0.8);
    }
    
    /* Professional Font Rendering */
    .stApp * {
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
    }
    
    /* Targeted visibility fixes for dark background elements */
    .stApp .element-container:not(.prediction-card):not(.upload-section) p,
    .stApp .element-container:not(.prediction-card):not(.upload-section) div:not([style*="background"]) {
        color: #cbd5e1 !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Ensure proper contrast for different backgrounds */
    .stApp [style*="background: rgba(248, 250, 252"] *,
    .stApp [style*="background: rgba(255, 255, 255"] *,
    .stApp [style*="background: white"] * {
        color: #1e293b !important;
        text-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pneumonia_model.keras')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Optimized preprocessing using CLAHE
def preprocess_image_clahe(image, target_size=(224, 224)):
    """
    Enhanced preprocessing for better pneumonia detection
    Reduced false negatives with gentler CLAHE and additional enhancement
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Gentler CLAHE to preserve subtle pneumonia signs
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    
    # Additional contrast enhancement for better pneumonia visibility
    img_enhanced = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    
    # Histogram equalization for additional enhancement
    img_eq = cv2.equalizeHist(img_gray)
    
    # Combine CLAHE and histogram equalization (70% CLAHE, 30% histogram eq)
    img_combined = cv2.addWeighted(img_enhanced, 0.7, img_eq, 0.3, 0)
    
    img_rgb = np.stack([img_combined, img_combined, img_combined], axis=-1)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
    img_normalized = img_resized.astype('float32') / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)
    
    return img_final

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Advanced threshold configuration for false negative reduction
PNEUMONIA_THRESHOLD_CONSERVATIVE = 0.3  # Very sensitive - catches more cases
PNEUMONIA_THRESHOLD_STANDARD = 0.4      # Balanced sensitivity
PNEUMONIA_THRESHOLD_STRICT = 0.5        # Original threshold

# Dynamic thresholding based on confidence distribution
def get_dynamic_threshold(confidence_score):
    """
    Dynamic threshold based on confidence patterns
    Lower threshold for borderline cases to reduce false negatives
    """
    if confidence_score < 0.2:
        return PNEUMONIA_THRESHOLD_CONSERVATIVE  # Very low confidence - be very sensitive
    elif confidence_score < 0.4:
        return PNEUMONIA_THRESHOLD_CONSERVATIVE  # Low confidence - be sensitive
    elif confidence_score < 0.7:
        return PNEUMONIA_THRESHOLD_STANDARD      # Medium confidence - balanced
    else:
        return PNEUMONIA_THRESHOLD_STRICT        # High confidence - can be stricter

# Threshold tuning options for user selection
THRESHOLD_OPTIONS = {
    "Conservative (0.3)": PNEUMONIA_THRESHOLD_CONSERVATIVE,
    "Balanced (0.4)": PNEUMONIA_THRESHOLD_STANDARD, 
    "Strict (0.5)": PNEUMONIA_THRESHOLD_STRICT,
    "Dynamic": "dynamic"
}

def get_prediction_analysis(confidence, result):
    """
    Analyze prediction confidence and provide medical warnings
    """
    warnings = []
    recommendations = []
    
    if result == "Normal" and confidence < 0.8:
        warnings.append("⚠️ **Low Confidence Normal**: Consider medical review for confirmation")
        recommendations.append("Recommend clinical correlation with symptoms")
    
    if result == "Pneumonia" and confidence < 0.7:
        warnings.append("⚠️ **Uncertain Pneumonia**: Clinical correlation strongly recommended")
        recommendations.append("Suggest radiologist review for confirmation")
    
    if 0.35 < confidence < 0.65:
        warnings.append("🔍 **Borderline Case**: Expert radiologist review recommended")
        recommendations.append("Consider additional imaging or clinical assessment")
    
    if result == "Normal" and 0.4 < confidence < 0.6:
        warnings.append("🚨 **Potential False Negative Risk**: High priority for medical review")
        recommendations.append("Do not rule out pneumonia based on AI alone")
    
    # Risk assessment
    if result == "Pneumonia":
        if confidence > 0.8:
            risk_level = "High Confidence"
            risk_color = "🔴"
        elif confidence > 0.6:
            risk_level = "Moderate Confidence"
            risk_color = "🟡"
        else:
            risk_level = "Low Confidence"
            risk_color = "🟠"
    else:
        if confidence > 0.8:
            risk_level = "High Confidence"
            risk_color = "🟢"
        elif confidence > 0.6:
            risk_level = "Moderate Confidence"
            risk_color = "🟡"
        else:
            risk_level = "Low Confidence - Review Needed"
            risk_color = "🟠"
    
    return {
        'warnings': warnings,
        'recommendations': recommendations,
        'risk_level': risk_level,
        'risk_color': risk_color
    }

# MongoDB Configuration
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/"  # Default local MongoDB

# Option 1: Use existing comprehensive database
DATABASE_NAME = "pneum_project"  # Your existing database with comprehensive data
COLLECTION_NAME = "predictions"

# Option 2: Use simple database (uncomment to switch)
# DATABASE_NAME = "pneumonia_detection"  # Simple database
# COLLECTION_NAME = "predictions"

def log_to_mongodb(prediction_data):
    """Log prediction to MongoDB database (compatible with existing structure)"""
    if not MONGODB_AVAILABLE:
        return False
    
    try:
        client = MongoClient(MONGODB_CONNECTION_STRING, serverSelectionTimeoutMS=2000)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Enhanced data structure to match existing format
        enhanced_data = {
            '_id': f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'user_id': 'anonymous',  # Since we removed patient info
            'patient_name': 'Anonymous',  # Anonymized
            'patient_age': None,
            'patient_gender': None,
            'prediction_result': prediction_data.get('result', 'Unknown'),
            'confidence_score': prediction_data.get('confidence_percent', 0) / 100.0,
            'image_filename': prediction_data.get('filename', 'unknown.jpg'),
            'image_size': None,
            'preprocessing_method': 'CLAHE',
            'model_version': 'v1.0',
            'analysis_data': {
                'confidence_level': prediction_data.get('confidence_level', 'Unknown'),
                'raw_prediction': prediction_data.get('raw_prediction', 0),
                'gradcam_available': True
            },
            'clinical_notes': 'AI-generated analysis - anonymized',
            'timestamp': datetime.now(),
            'status': 'completed'
        }
        
        # Insert the document
        result = collection.insert_one(enhanced_data)
        client.close()
        
        return result.inserted_id is not None
        
    except Exception as e:
        st.sidebar.warning(f"MongoDB logging failed: {str(e)}")
        return False

def get_mongodb_stats():
    """Get statistics from MongoDB"""
    if not MONGODB_AVAILABLE:
        return None
    
    try:
        client = MongoClient(MONGODB_CONNECTION_STRING, serverSelectionTimeoutMS=2000)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        total_predictions = collection.count_documents({})
        # Use the existing field name from your database
        pneumonia_cases = collection.count_documents({"prediction_result": "Pneumonia"})
        normal_cases = collection.count_documents({"prediction_result": "Normal"})
        
        client.close()
        
        return {
            "total": total_predictions,
            "pneumonia": pneumonia_cases,
            "normal": normal_cases
        }
        
    except Exception as e:
        return None

# PDF Report Generation Functions
def create_analysis_report(prediction_result, confidence, original_image, heatmap_image=None):
    """Generate professional analysis PDF report (anonymized)"""
    
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    # Title
    story.append(Paragraph("CHEST X-RAY AI ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 12))
    
    # Analysis Information Table (anonymized)
    story.append(Paragraph("ANALYSIS INFORMATION", header_style))
    
    analysis_data = [
        ['Analysis Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Analysis Type:', 'Pneumonia Detection'],
        ['AI Model:', 'Deep Learning CNN'],
        ['Report ID:', f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
    ]
    
    analysis_table = Table(analysis_data, colWidths=[2*inch, 3*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(analysis_table)
    story.append(Spacer(1, 20))
    
    # AI Analysis Results
    story.append(Paragraph("AI ANALYSIS RESULTS", header_style))
    
    # Prediction result with color coding
    result_color = colors.red if prediction_result == "Pneumonia" else colors.green
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=normal_style,
        fontSize=12,
        textColor=result_color,
        spaceAfter=6
    )
    
    story.append(Paragraph(f"<b>Primary Finding:</b> {prediction_result}", result_style))
    story.append(Paragraph(f"<b>Confidence Level:</b> {confidence:.1f}%", normal_style))
    
    # Risk assessment
    if prediction_result == "Pneumonia":
        if confidence > 90:
            risk_level = "HIGH RISK"
            risk_color = colors.red
        elif confidence > 70:
            risk_level = "MODERATE RISK"
            risk_color = colors.orange
        else:
            risk_level = "LOW-MODERATE RISK"
            risk_color = colors.orange
    else:
        risk_level = "LOW RISK"
        risk_color = colors.green
    
    risk_style = ParagraphStyle('RiskStyle', parent=normal_style, textColor=risk_color, fontSize=11)
    story.append(Paragraph(f"<b>Risk Assessment:</b> {risk_level}", risk_style))
    story.append(Spacer(1, 15))
    
    # Clinical Notes
    story.append(Paragraph("CLINICAL NOTES", header_style))
    
    clinical_notes = []
    if prediction_result == "Pneumonia":
        if confidence > 80:
            clinical_notes.append("HIGH CONFIDENCE PNEUMONIA DETECTION: AI model indicates pneumonia with high confidence.")
            clinical_notes.append("RECOMMENDATION: Clinical correlation with symptoms, vital signs, and laboratory findings advised.")
        else:
            clinical_notes.append("MODERATE CONFIDENCE PNEUMONIA DETECTION: AI model suggests possible pneumonia.")
            clinical_notes.append("RECOMMENDATION: Further imaging or clinical follow-up recommended to confirm findings.")
    else:
        clinical_notes.append("NORMAL CHEST X-RAY: AI analysis indicates normal findings.")
        if confidence < 80:
            clinical_notes.append("RECOMMENDATION: If clinical suspicion remains high, consider repeat imaging or additional studies.")
    
    clinical_notes.append("NOTE: This AI analysis is for educational and research purposes only. Final diagnosis requires qualified medical interpretation.")
    
    for note in clinical_notes:
        clean_note = note.replace("**", "").replace("*", "")
        story.append(Paragraph(f"• {clean_note}", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    story.append(Paragraph("This report was generated by AI-powered chest X-ray analysis system.", footer_style))
    story.append(Paragraph("For educational and research purposes. Requires qualified medical interpretation.", footer_style))
    story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    
    # Build PDF
    doc.build(story)
    
    # Get the value of the BytesIO buffer and return it
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def create_download_link(pdf_data, filename):
    """Create a download link for the PDF report"""
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" '
    href += 'style="display: inline-block; text-decoration: none; '
    href += 'background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; '
    href += 'border-radius: 0.75rem; padding: 0.75rem 2rem; font-weight: 600; cursor: pointer; '
    href += 'box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">'
    href += '📄 Download Analysis Report (PDF)</a>'
    return href

def create_html_email_report(prediction_result, confidence, analysis_timestamp, heatmap_available=False):
    """Create HTML email report content"""
    
    # Determine result styling
    if prediction_result == "Pneumonia":
        result_color = "#ef4444"
        result_bg = "#fef2f2"
        risk_level = "HIGH RISK" if confidence > 80 else "MODERATE RISK"
        risk_color = "#dc2626" if confidence > 80 else "#f59e0b"
    else:
        result_color = "#10b981"
        result_bg = "#ecfdf5"
        risk_level = "LOW RISK"
        risk_color = "#059669"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pneumonia Detection Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #374151;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9fafb;
            }}
            .header {{
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
                padding: 30px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 700;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 16px;
            }}
            .report-card {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                border-left: 5px solid {result_color};
            }}
            .result-section {{
                background: {result_bg};
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border: 1px solid {result_color}33;
            }}
            .result-title {{
                color: {result_color};
                font-size: 24px;
                font-weight: 700;
                margin: 0 0 10px 0;
            }}
            .confidence-score {{
                font-size: 20px;
                font-weight: 600;
                color: #1f2937;
                margin: 10px 0;
            }}
            .risk-badge {{
                display: inline-block;
                background: {risk_color};
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 14px;
                margin: 10px 0;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 20px 0;
            }}
            .info-item {{
                background: #f8fafc;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .info-label {{
                font-weight: 600;
                color: #4b5563;
                font-size: 14px;
                margin-bottom: 5px;
            }}
            .info-value {{
                color: #1f2937;
                font-size: 16px;
                font-weight: 500;
            }}
            .disclaimer {{
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 8px;
                padding: 20px;
                margin: 25px 0;
            }}
            .disclaimer h3 {{
                color: #92400e;
                margin: 0 0 10px 0;
                font-size: 18px;
            }}
            .disclaimer p {{
                color: #78350f;
                margin: 0;
                font-weight: 500;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #6b7280;
                font-size: 14px;
                border-top: 1px solid #e5e7eb;
                margin-top: 30px;
            }}
            .features {{
                background: #f0f9ff;
                border: 1px solid #0ea5e9;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
            .features h3 {{
                color: #0c4a6e;
                margin: 0 0 15px 0;
            }}
            .feature-list {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            .feature-list li {{
                color: #075985;
                margin: 8px 0;
                padding-left: 20px;
                position: relative;
            }}
            .feature-list li:before {{
                content: "✓";
                position: absolute;
                left: 0;
                color: #0ea5e9;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🫁 Pneumonia Detection Report</h1>
            <p>AI-Powered Chest X-Ray Analysis System</p>
        </div>
        
        <div class="report-card">
            <h2 style="color: #1f2937; margin-top: 0;">Dear Healthcare Professional,</h2>
            <p>Please find below the AI-powered pneumonia detection analysis results:</p>
            
            <div class="result-section">
                <div class="result-title">PREDICTION RESULT: {prediction_result}</div>
                <div class="confidence-score">CONFIDENCE SCORE: {confidence:.1f}%</div>
                <div class="risk-badge">{risk_level}</div>
            </div>
            
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Analysis Date</div>
                    <div class="info-value">{analysis_timestamp}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">AI Model</div>
                    <div class="info-value">Deep Learning CNN</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Accuracy Rate</div>
                    <div class="info-value">>90% Validated</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Explainability</div>
                    <div class="info-value">{"Grad-CAM Available" if heatmap_available else "Standard Analysis"}</div>
                </div>
            </div>
            
            <p>This AI-powered analysis was performed on a chest X-ray image using our pneumonia detection system.</p>
        </div>
        
        <div class="features">
            <h3>🔬 Analysis Features</h3>
            <ul class="feature-list">
                <li>Deep Learning CNN with >90% accuracy</li>
                <li>CLAHE preprocessing for enhanced image quality</li>
                <li>Grad-CAM explainable AI visualization</li>
                <li>Real-time prediction with confidence scoring</li>
                <li>Professional medical report generation</li>
            </ul>
        </div>
        
        <div class="disclaimer">
            <h3>⚠️ IMPORTANT DISCLAIMER</h3>
            <p>This analysis is for screening purposes only and should not replace professional medical judgment. 
            Please consult with a qualified radiologist or medical professional for definitive diagnosis.</p>
        </div>
        
        <div style="text-align: center; margin: 30px 0;">
            <p style="font-size: 16px; color: #374151; margin: 0;">Thank you for using our Pneumonia Detection System.</p>
            <p style="font-size: 18px; font-weight: 600; color: #1f2937; margin: 10px 0 0 0;">
                Best regards,<br>
                AI Medical Assistant<br>
                Pneumonia Detection System
            </p>
        </div>
        
        <div class="footer">
            <p>---</p>
            <p>Report generated on: {analysis_timestamp}</p>
            <p>This is an automated email from the Pneumonia Detection AI System</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def send_email_report(recipient_email, prediction_result, confidence, analysis_timestamp, heatmap_available=False):
    """Send HTML email report using environment variables"""
    if not EMAIL_AVAILABLE:
        return False, "Email functionality not available"
    
    if not SMTP_USER or not SMTP_PASSWORD:
        return False, "Email service not configured"
    
    try:
        # Email configuration from environment
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_USER
        msg['To'] = recipient_email
        msg['Subject'] = f"Pneumonia Detection Analysis Report - {prediction_result} ({confidence:.1f}% confidence)"
        
        # Create HTML content
        html_content = create_html_email_report(
            prediction_result, confidence, analysis_timestamp, heatmap_available
        )
        
        # Create plain text version
        text_content = f"""
Dear Healthcare Professional,

Pneumonia Detection Analysis Report

PREDICTION RESULT: {prediction_result}
CONFIDENCE SCORE: {confidence:.1f}%
Analysis Date: {analysis_timestamp}

This AI-powered analysis was performed on a chest X-ray image using our pneumonia detection system.

IMPORTANT DISCLAIMER:
This analysis is for screening purposes only and should not replace professional medical judgment.
Please consult with a qualified radiologist or medical professional for definitive diagnosis.

Thank you for using our Pneumonia Detection System.

Best regards,
AI Medical Assistant
Pneumonia Detection System

---
Report generated on: {analysis_timestamp}
        """
        
        # Attach both versions
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USER, recipient_email, text)
        server.quit()
        
        return True, f"Email report sent successfully to {recipient_email}"
        
    except Exception as e:
        return False, f"Email sending failed: {str(e)}"

def main():
    load_css()
    
    # Sidebar with statistics and info
    with st.sidebar:
        st.markdown("### 📊 System Statistics")
        
        # MongoDB statistics
        if MONGODB_AVAILABLE:
            stats = get_mongodb_stats()
            if stats:
                st.metric("Total Predictions", stats['total'])
                st.metric("Pneumonia Cases", stats['pneumonia'])
                st.metric("Normal Cases", stats['normal'])
            else:
                st.info("📊 Database connection unavailable")
        
        # Local session statistics
        if st.session_state.prediction_history:
            local_total = len(st.session_state.prediction_history)
            local_pneumonia = sum(1 for p in st.session_state.prediction_history if 'Pneumonia' in p['result'])
            local_normal = local_total - local_pneumonia
            
            st.markdown("#### 📱 Session Statistics")
            st.metric("Session Predictions", local_total)
            st.metric("Session Pneumonia", local_pneumonia)
            st.metric("Session Normal", local_normal)
        
        st.markdown("---")
        st.markdown("### 🔧 System Info")
        st.info("🎯 **Model Accuracy**: >90%")
        st.info("🧠 **AI Model**: Deep Learning CNN")
        st.info("🔍 **Explainability**: Grad-CAM")
        
        if MONGODB_AVAILABLE:
            st.success("📝 **Database**: Connected")
        else:
            st.warning("📝 **Database**: Local only")
        
        if EMAIL_AVAILABLE:
            st.success("📧 **Email**: Available")
        else:
            st.warning("📧 **Email**: Not configured")
    
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <span class="medical-icon" style="font-size: 4rem; margin-right: 1rem;">🫁</span>
            <div>
                <h1 style="color: white; font-size: 2.8rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.4); letter-spacing: -0.025em;">
                    Pneumonia Detection
                </h1>
                <h2 style="color: white; font-size: 1.6rem; font-weight: 500; margin: 0; opacity: 0.9; letter-spacing: 0.025em;">
                    AI Dashboard
                </h2>
            </div>
        </div>
        <p style="color: #e2e8f0; font-size: 1.1rem; margin: 1rem 0; font-weight: 400; text-shadow: 0 1px 2px rgba(0,0,0,0.3); line-height: 1.5;">
            🤖 AI-Powered Chest X-Ray Analysis with Explainable AI
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem;">
            <div class="accuracy-badge">
                🎯 >90% Accuracy
            </div>
            <div class="accuracy-badge" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
                🔍 Grad-CAM Explainability
            </div>
            <div class="accuracy-badge" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);">
                📊 Real-time Analysis
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 1rem; backdrop-filter: blur(10px);">
            <p style="color: #f1f5f9; font-size: 0.9rem; margin: 0; font-weight: 400; opacity: 0.9;">
                🩺 Professional-grade AI system for educational and research purposes
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        st.error("❌ Could not load the pneumonia detection model. Please ensure 'pneumonia_model.keras' is in the current directory.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <span style="font-size: 3rem; margin-right: 1rem;">📤</span>
                <div>
                    <h3 style="color: #0f172a; margin: 0; font-size: 1.5rem; font-weight: 600; letter-spacing: -0.025em;">
                        Upload Chest X-Ray
                    </h3>
                    <p style="color: #1e293b; margin: 0; font-size: 1rem; font-weight: 600; line-height: 1.5;">
                        Drag & drop or click to select
                    </p>
                </div>
            </div>
            <div style="padding: 1rem; background: rgba(59, 130, 246, 0.05); border-radius: 0.75rem; margin-top: 1rem;">
                <p style="color: #1e293b; font-size: 1rem; margin: 0; font-weight: 600; line-height: 1.6;">
                    📋 Supported formats: JPG, PNG, JPEG<br>
                    🔬 AI will analyze for pneumonia indicators<br>
                    ⚡ Results in seconds with explainable AI
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom styling for file uploader
        st.markdown("""
        <style>
        .stFileUploader > div > div > div > div {
            color: #1e293b !important;
            font-weight: 700 !important;
            font-size: 1.1em !important;
        }
        .stFileUploader > div > div > div > button {
            color: #1e293b !important;
            font-weight: 700 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded X-Ray Image", use_column_width=True)
            
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the X-ray image and generating Grad-CAM..."):
                    processed_image = preprocess_image_clahe(image)
                    
                    # Use the new Grad-CAM function that returns both prediction and heatmap
                    gradcam_result = generate_gradcam_with_prediction(model, processed_image)
                    
                    result = gradcam_result['prediction_result']
                    confidence_percent = gradcam_result['confidence']
                    
                    # Get prediction analysis with warnings
                    analysis = get_prediction_analysis(confidence_percent/100, result)
                    
                    if result == "Pneumonia":
                        card_class = "pneumonia-card"
                        icon = "🚨"
                    else:
                        card_class = "normal-card"
                        icon = "✅"
                    
                    if confidence_percent >= 90:
                        conf_class = "confidence-high"
                        conf_icon = "🟢"
                        conf_level = "Very High"
                    elif confidence_percent >= 75:
                        conf_class = "confidence-high"
                        conf_icon = "🟢"
                        conf_level = "High"
                    elif confidence_percent >= 60:
                        conf_class = "confidence-medium"
                        conf_icon = "🟡"
                        conf_level = "Medium"
                    else:
                        conf_class = "confidence-low"
                        conf_icon = "🔴"
                        conf_level = "Low"
                    
                    st.session_state.current_prediction = {
                        'result': result,
                        'confidence': confidence_percent,
                        'conf_class': conf_class,
                        'conf_icon': conf_icon,
                        'conf_level': conf_level,
                        'card_class': card_class,
                        'icon': icon,
                        'image': image,
                        'processed_image': processed_image,
                        'gradcam_base64': gradcam_result['gradcam_heatmap'],
                        'analysis': analysis
                    }
                    
                    # Add to local history
                    prediction_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'result': result,
                        'confidence': f"{confidence_percent:.1f}%"
                    }
                    st.session_state.prediction_history.append(prediction_record)
                    
                    # Log to MongoDB
                    mongodb_data = {
                        'filename': uploaded_file.name,
                        'result': result,
                        'confidence_percent': confidence_percent,
                        'confidence_level': conf_level,
                        'raw_prediction': confidence_percent / 100.0,
                        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Log to MongoDB and show confirmation
                    mongodb_success = log_to_mongodb(mongodb_data)
                    
                    if mongodb_success:
                        # Show detailed success message
                        st.success("✅ **Prediction Stored Successfully in Database!**")
                        
                        # Show what was stored in an expandable section
                        with st.expander("📊 View Stored Data", expanded=False):
                            st.json({
                                "Result": result,
                                "Confidence": f"{confidence_percent:.1f}%",
                                "Filename": uploaded_file.name,
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Database": "pneum_project",
                                "Collection": "predictions",
                                "Status": "Successfully Stored"
                            })
                        
                        st.sidebar.success("📝 Prediction logged to database")
                    else:
                        st.warning("⚠️ **Database Storage Failed** - Prediction stored locally only.")
                        st.info("💡 Check MongoDB connection or contact administrator.")
                        st.sidebar.info("📝 Prediction stored locally only")
    
    with col2:
        if 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction
            
            st.markdown(f"""
            <div class="prediction-card {pred['card_class']}">
                <h3 style="margin: 0; color: #374151; display: flex; align-items: center; font-size: 1.3rem;">
                    <span style="margin-right: 0.5rem; font-size: 2rem;">{pred['icon']}</span>
                    Prediction Result
                </h3>
                <div style="margin-top: 1.5rem;">
                    <h2 style="margin: 0; color: #0f172a; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.025em;">
                        {pred['result']}
                    </h2>
                    <p style="margin: 1rem 0 0 0; color: #6b7280; font-size: 1.2rem;">
                        Confidence: <span class="{pred['conf_class']}">{pred['confidence']:.1f}%</span>
                        <span style="margin-left: 0.5rem;">{pred['conf_icon']} {pred['conf_level']}</span>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display confidence level
            if pred['conf_level'] in ['Very High', 'High']:
                st.success("✅ **High Confidence Prediction**: The AI model is very confident in this diagnosis.")
            elif pred['conf_level'] == 'Medium':
                st.info("ℹ️ **Medium Confidence**: Consider additional medical evaluation for confirmation.")
            else:
                st.warning("⚠️ **Low Confidence**: This prediction has low confidence. Medical professional consultation recommended.")
            
            # Display analysis warnings and recommendations
            if 'analysis' in pred:
                analysis = pred['analysis']
                
                # Show warnings
                if analysis['warnings']:
                    st.markdown("#### 🚨 Medical Analysis Warnings")
                    for warning in analysis['warnings']:
                        st.warning(warning)
                
                # Show recommendations
                if analysis['recommendations']:
                    st.markdown("#### 💡 Clinical Recommendations")
                    for rec in analysis['recommendations']:
                        st.info(f"• {rec}")
                
                # Show risk assessment
                st.markdown(f"#### {analysis['risk_color']} Risk Assessment: {analysis['risk_level']}")
            
            # Always show medical disclaimer for pneumonia cases
            if pred['result'] == "Pneumonia":
                st.error("🏥 **IMPORTANT**: This AI analysis is for screening only. Immediate medical consultation is recommended for suspected pneumonia cases.")
            elif pred['result'] == "Normal" and pred['confidence'] < 80:
                st.warning("🏥 **CAUTION**: Low confidence normal prediction. Do not rule out pneumonia without clinical evaluation.")
            
            # --- GRAD-CAM EXPLAINABLE AI SECTION ---
            st.markdown("### 🔥 Grad-CAM Explainable AI")
            
            # Decode base64 heatmap for display
            import base64
            import io
            
            heatmap_data = base64.b64decode(pred['gradcam_base64'])
            heatmap_image = Image.open(io.BytesIO(heatmap_data))
            
            # Display images side by side
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(pred['image'], caption="📸 Original X-Ray", use_column_width=True)
            with img_col2:
                st.image(heatmap_image, caption="🔥 Grad-CAM Heatmap", use_column_width=True)
            
            st.success("🔍 **Grad-CAM Analysis**: Red/yellow areas highlight regions the AI focused on for diagnosis.")
            st.info("💡 **Interpretation**: Brighter colors indicate higher AI attention. This explainable AI helps understand the decision-making process.")
            
            # Store heatmap for PDF generation
            st.session_state.current_prediction['heatmap_image'] = heatmap_image
            
            # Email Report Generation Section
            st.markdown("### 📧 Email Report Generation")
            
            if EMAIL_AVAILABLE:
                st.info("� **Email Service**: Professional Medical Report Delivery")
                
                # Email input field
                email_address = st.text_input(
                    "📧 Recipient Email Address", 
                    placeholder="Enter email address to receive the analysis report",
                    help="Enter the email address where you want to receive the detailed analysis report"
                )
                
                if email_address:
                    # Validate email format
                    import re
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    
                    if re.match(email_pattern, email_address):
                        st.success(f"✅ Valid email address: {email_address}")
                        
                        if st.button("📧 Send Analysis Report via Email", key="send_email", use_container_width=True):
                            with st.spinner("📧 Sending email report..."):
                                try:
                                    # Check if heatmap is available
                                    heatmap_available = 'heatmap_image' in pred
                                    
                                    # Get analysis timestamp
                                    analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    # Send email report
                                    success, message = send_email_report(
                                        recipient_email=email_address,
                                        prediction_result=pred['result'],
                                        confidence=pred['confidence'],
                                        analysis_timestamp=analysis_timestamp,
                                        heatmap_available=heatmap_available
                                    )
                                    
                                    if success:
                                        st.success(f"✅ {message}")
                                        st.balloons()
                                        
                                        # Show email preview
                                        with st.expander("📧 Email Preview", expanded=False):
                                            st.markdown("**Subject:** " + f"Pneumonia Detection Analysis Report - {pred['result']} ({pred['confidence']:.1f}% confidence)")
                                            st.markdown("**From:** Medical AI Analysis System")
                                            st.markdown("**To:** " + email_address)
                                            st.markdown("**Content:** Professional HTML email with analysis results, confidence scores, and medical disclaimers")
                                    else:
                                        st.error(f"❌ {message}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error sending email: {str(e)}")
                    else:
                        st.error("❌ Please enter a valid email address")
                else:
                    st.info("👆 Enter an email address above to send the analysis report")
            else:
                st.error("❌ Email functionality not available. Please contact administrator.")
                st.info("💡 Email service requires proper configuration by system administrator")
        
        else:
            st.markdown("""
            <div class="prediction-card">
                <h3 style="margin: 0; color: #6b7280; text-align: center; font-size: 1.3rem;">
                    📊 Prediction Results
                </h3>
                <p style="text-align: center; color: #1e293b; margin-top: 1rem; font-size: 1.1rem; font-weight: 600;">
                    Upload an X-ray image to see AI analysis results
                </p>
                <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: rgba(248, 250, 252, 0.2); border-radius: 0.5rem; border: 1px solid rgba(148, 163, 184, 0.4);">
                    <p style="margin: 0; color: #0f172a; font-weight: 700; font-size: 1.1rem;">Model Performance:</p>
                    <p style="margin: 0.5rem 0 0 0; color: #1e293b; font-weight: 600; font-size: 1rem;">✅ >90% accuracy on both Pneumonia & Normal cases</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📋 Prediction History")
        
        df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": "Date & Time",
                "filename": "File Name", 
                "result": "Result",
                "confidence": "Confidence"
            }
        )
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            if 'current_prediction' in st.session_state:
                del st.session_state.current_prediction
            st.rerun()

if __name__ == "__main__":
    main()