import os
import smtplib
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, List
import json

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("pymongo not available. MongoDB logging will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmailSender:
    def __init__(self):
        """Initialize email sender with SMTP settings and MongoDB connection."""
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.default_password = os.getenv("SMTP_PASSWORD")
        
        # MongoDB connection setup
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/pneum_project")
        self.mongodb_client = None
        self.mongodb_db = None
        self.mongodb_collection = None
        self.mongodb_available = False
        self.fallback_reports = []  # In-memory fallback storage
        
        # Initialize MongoDB connection
        self._init_mongodb()
        
        logger.info("EmailSender initialized successfully")

    def _init_mongodb(self):
        """Initialize MongoDB connection with error handling."""
        if not MONGODB_AVAILABLE:
            logger.warning("MongoDB not available - pymongo not installed")
            return
        
        try:
            # Connect to MongoDB with timeout
            self.mongodb_client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.mongodb_client.admin.command('ping')
            
            # Get database and collection
            self.mongodb_db = self.mongodb_client.get_database("pneum_project")
            self.mongodb_collection = self.mongodb_db.get_collection("reports")
            
            self.mongodb_available = True
            logger.info(f"MongoDB connected successfully to {self.mongodb_uri}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"MongoDB connection failed: {e}")
            logger.info("Falling back to in-memory storage for report logging")
            self.mongodb_available = False
            self.mongodb_client = None
            self.mongodb_db = None
            self.mongodb_collection = None
        except Exception as e:
            logger.error(f"Unexpected MongoDB error: {e}")
            self.mongodb_available = False

    def log_report(self, receiver_email: str, prediction_result: str, confidence_score: float) -> Dict[str, any]:
        """
        Log email report to MongoDB or fallback storage.
        
        Args:
            receiver_email: Email address of the recipient
            prediction_result: The prediction result ("Pneumonia" or "Normal")
            confidence_score: Confidence score as a percentage (0-100)
            
        Returns:
            dict: Result with success status and message
        """
        # Create report document
        report_doc = {
            "receiver_email": receiver_email,
            "prediction_result": prediction_result,
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.mongodb_available and self.mongodb_collection is not None:
            try:
                # Insert into MongoDB
                result = self.mongodb_collection.insert_one(report_doc)
                logger.info(f"Report logged to MongoDB with ID: {result.inserted_id}")
                return {
                    "success": True,
                    "message": f"Report logged to MongoDB successfully",
                    "storage": "mongodb",
                    "document_id": str(result.inserted_id)
                }
            except Exception as e:
                logger.error(f"Failed to log to MongoDB: {e}")
                # Fall through to fallback storage
        
        # Fallback to in-memory storage
        try:
            self.fallback_reports.append(report_doc)
            logger.info(f"Report logged to fallback storage (total reports: {len(self.fallback_reports)})")
            return {
                "success": True,
                "message": f"Report logged to fallback storage (MongoDB unavailable)",
                "storage": "fallback",
                "total_reports": len(self.fallback_reports)
            }
        except Exception as e:
            logger.error(f"Failed to log to fallback storage: {e}")
            return {
                "success": False,
                "error": f"Failed to log report: {e}",
                "storage": "none"
            }

    def get_reports_count(self) -> Dict[str, any]:
        """Get the count of logged reports."""
        if self.mongodb_available and self.mongodb_collection is not None:
            try:
                count = self.mongodb_collection.count_documents({})
                return {
                    "success": True,
                    "count": count,
                    "storage": "mongodb"
                }
            except Exception as e:
                logger.error(f"Failed to get MongoDB count: {e}")
        
        # Fallback count
        return {
            "success": True,
            "count": len(self.fallback_reports),
            "storage": "fallback"
        }

    def validate_email(self, email: str) -> bool:
        """
        Validate email format with enhanced validation.
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if email is valid, False otherwise
        """
        if not email or not isinstance(email, str):
            return False
        
        # Trim whitespace
        email = email.strip()
        
        if not email:
            return False
        
        # Enhanced regex pattern that handles more edge cases
        # This pattern is more restrictive and catches common invalid formats
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9._+-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,63}$'
        
        # Check basic regex match
        if not re.match(pattern, email):
            return False
        
        # Additional checks for edge cases
        # Check for consecutive dots
        if '..' in email:
            return False
        
        # Check for dots at start/end of local part
        local_part, domain_part = email.split('@', 1)
        if local_part.startswith('.') or local_part.endswith('.'):
            return False
        
        # Check for dots at start/end of domain part
        if domain_part.startswith('.') or domain_part.endswith('.'):
            return False
        
        # Check TLD length (2-63 characters)
        tld = domain_part.split('.')[-1]
        if len(tld) < 2 or len(tld) > 63:
            return False
        
        return True

    def generate_email_content(self, prediction_result: Dict) -> Tuple[str, str]:
        """
        Generate professional email subject and body using templates.
        
        Args:
            prediction_result: Dictionary containing prediction details
            
        Returns:
            tuple: (subject, body) - Email subject and body
        """
        result = prediction_result.get('result', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        patient_name = prediction_result.get('patient_name', 'Anonymous')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Professional subject line
        subject = f"Pneumonia Detection Analysis - {result} - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Professional email body template
        body = f"""Dear Healthcare Professional,

Please find below the AI-assisted pneumonia detection analysis results:

PATIENT INFORMATION:
• Patient Name: {patient_name}
• Analysis Date: {timestamp}
• Analysis ID: PNA-{datetime.now().strftime('%Y%m%d%H%M%S')}

ANALYSIS RESULTS:
• Detection Result: {result}
• Confidence Level: {confidence:.1f}%
• Analysis Method: AI-powered chest X-ray screening

CLINICAL INTERPRETATION:
{self._get_clinical_interpretation(result, confidence)}

IMPORTANT DISCLAIMERS:
• This AI analysis is intended for screening and triage purposes only
• Professional medical interpretation and clinical correlation are required
• Final diagnosis should be based on comprehensive clinical assessment
• This system is designed to assist, not replace, clinical judgment

NEXT STEPS:
{self._get_next_steps(result, confidence)}

For questions regarding this analysis, please contact your IT support team.

Best regards,
Pneumonia Detection System
Automated Medical AI Assistant

---
This is an automated message generated by the Pneumonia Detection System.
Report generated on: {timestamp}
        """
        
        return subject, body

    def _get_clinical_interpretation(self, result: str, confidence: float) -> str:
        """Generate clinical interpretation based on results."""
        if result.upper() == 'PNEUMONIA':
            if confidence >= 90:
                return "• High confidence detection of pneumonia-like patterns\n• Recommend immediate clinical review and correlation\n• Consider urgent radiologist consultation"
            elif confidence >= 70:
                return "• Moderate confidence detection of pneumonia-like patterns\n• Recommend clinical review within standard timeframe\n• Consider radiologist review if clinically indicated"
            else:
                return "• Low-moderate confidence detection\n• Recommend clinical correlation and possible repeat imaging\n• Consider additional diagnostic workup if symptoms persist"
        else:
            if confidence >= 90:
                return "• High confidence normal chest X-ray appearance\n• No obvious signs of pneumonia detected\n• Continue standard clinical assessment"
            elif confidence >= 70:
                return "• Moderate confidence normal appearance\n• Low probability of pneumonia\n• Clinical correlation recommended"
            else:
                return "• Uncertain analysis results\n• Recommend professional radiologist review\n• Consider repeat imaging if clinically indicated"

    def _get_next_steps(self, result: str, confidence: float) -> str:
        """Generate next steps based on results."""
        if result.upper() == 'PNEUMONIA':
            return "• Immediate clinical assessment recommended\n• Consider antibiotic therapy if clinically appropriate\n• Monitor patient response and vital signs\n• Follow institutional pneumonia protocols"
        else:
            return "• Continue standard clinical monitoring\n• Reassess if symptoms worsen or persist\n• Consider alternative diagnoses if clinically indicated\n• Follow up as per standard protocols"

    def send_email(
        self,
        sender_email: str,
        recipient: str,
        subject: str,
        body: str,
        attachment_path: Optional[str] = None,
        sender_password: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Send an email with optional attachment using dynamic sender.
        
        Args:
            sender_email: Email address of the sender
            recipient: Email address of the recipient
            subject: Email subject
            body: Email body (HTML or plain text)
            attachment_path: Path to file to attach (optional)
            sender_password: Password for sender email (optional, uses default if not provided)
            
        Returns:
            dict: Result with success status and message
        """
        # Validate email addresses
        if not self.validate_email(sender_email):
            return {
                "success": False,
                "error": f"Invalid sender email format: {sender_email}"
            }
        
        if not self.validate_email(recipient):
            return {
                "success": False,
                "error": f"Invalid recipient email format: {recipient}"
            }
        
        # Use provided password or default
        password = sender_password or self.default_password
        if not password:
            return {
                "success": False,
                "error": "No password provided for email authentication"
            }
            
        try:
            # Create message container
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Attach body
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach file if provided
            if attachment_path and os.path.exists(attachment_path):
                try:
                    with open(attachment_path, 'rb') as f:
                        part = MIMEApplication(
                            f.read(),
                            Name=os.path.basename(attachment_path)
                        )
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                    msg.attach(part)
                except Exception as e:
                    logger.warning(f"Failed to attach file {attachment_path}: {e}")
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
                
            logger.info(f"Email sent successfully from {sender_email} to {recipient}")
            return {
                "success": True,
                "message": f"Email sent successfully to {recipient}"
            }
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed for {sender_email}. Check email and password."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error sending email: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def send_email_report(self, receiver_email: str, prediction_result: str, confidence_score: float) -> Dict[str, any]:
        """
        Send a simple email report with prediction results and log to MongoDB.
        
        Args:
            receiver_email: Email address of the recipient
            prediction_result: The prediction result ("Pneumonia" or "Normal")
            confidence_score: Confidence score as a percentage (0-100)
            
        Returns:
            dict: Result with success status and message
        """
        # Get sender email from environment
        sender_email = os.getenv("SMTP_USER")
        if not sender_email:
            return {
                "success": False,
                "error": "SMTP_USER not found in environment variables"
            }
        
        # Validate receiver email
        if not self.validate_email(receiver_email):
            return {
                "success": False,
                "error": f"Invalid receiver email format: {receiver_email}"
            }
        
        # Generate simple email content
        subject = f"Pneumonia Detection Report - {prediction_result}"
        
        body = f"""Dear Healthcare Professional,

Pneumonia Detection Analysis Report:

PREDICTION RESULT: {prediction_result}
CONFIDENCE SCORE: {confidence_score:.1f}%

This AI-powered analysis was performed on a chest X-ray image using our pneumonia detection system.

IMPORTANT DISCLAIMER:
This analysis is for screening purposes only and should not replace professional medical judgment. 
Please consult with a qualified radiologist or medical professional for definitive diagnosis.

Thank you for using our Pneumonia Detection System.

Best regards,
AI Medical Assistant
Pneumonia Detection System

---
Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Send the email
        email_result = self.send_email(
            sender_email=sender_email,
            recipient=receiver_email,
            subject=subject,
            body=body,
            sender_password=self.default_password
        )
        
        # If email was sent successfully, log the report
        if email_result.get('success', False):
            # Log to MongoDB or fallback storage
            log_result = self.log_report(receiver_email, prediction_result, confidence_score)
            
            # Add logging information to the result
            email_result['logging'] = log_result
            
            # Update success message to include logging info
            if log_result.get('success', False):
                storage_type = log_result.get('storage', 'unknown')
                if storage_type == 'mongodb':
                    email_result['message'] += f" | Report logged to MongoDB"
                elif storage_type == 'fallback':
                    email_result['message'] += f" | Report logged to fallback storage"
            else:
                email_result['message'] += f" | Warning: Report logging failed"
        
        return email_result

# Create a global instance for easy import
email_sender = EmailSender()
