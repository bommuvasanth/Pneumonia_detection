from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
from datetime import datetime
import logging
from database import get_database, PneumoniaDatabase
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered pneumonia detection system with MongoDB integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    role: Optional[str] = "user"
    department: Optional[str] = None
    hospital: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    department: Optional[str]
    hospital: Optional[str]
    created_at: datetime
    is_active: bool

class PredictionRequest(BaseModel):
    user_id: Optional[str] = None
    patient_name: Optional[str] = "Anonymous"
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    sender_email: Optional[EmailStr] = None
    sender_password: Optional[str] = None
    recipient_email: Optional[EmailStr] = None
    send_notification: Optional[bool] = False

class PredictionResponse(BaseModel):
    id: str
    prediction_result: str
    confidence_score: float
    patient_name: str
    timestamp: datetime
    analysis_data: Optional[Dict] = None
    clinical_notes: Optional[List[str]] = None
    email_notification: Optional[Dict] = None

class MedicalReportCreate(BaseModel):
    prediction_id: str
    patient_info: Optional[Dict] = {}
    findings: str
    clinical_notes: List[str]
    radiologist_notes: Optional[str] = None
    report_type: Optional[str] = "ai_screening"
    status: Optional[str] = "draft"
    created_by: Optional[str] = None

class EmailNotificationRequest(BaseModel):
    sender_email: EmailStr
    sender_password: str
    recipient_email: EmailStr
    prediction_id: str
    include_attachment: Optional[bool] = True

class EmailNotificationResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Load the ML model on startup"""
    global model
    try:
        model = tf.keras.models.load_model('pneumonia_model.keras')
        # Test model with dummy input
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        logger.info("Pneumonia model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model = None

def get_db() -> PneumoniaDatabase:
    """Dependency to get database instance"""
    return get_database()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to (224, 224)
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert to RGB (3 channels)
        if len(img_resized.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif len(img_resized.shape) == 3 and img_resized.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
        else:  # Already RGB
            img_rgb = img_resized
        
        # Normalize pixel values (divide by 255)
        img_normalized = img_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        img_final = np.expand_dims(img_normalized, axis=0)
        
        return img_final
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def analyze_prediction(prediction_array: np.ndarray) -> Dict:
    """Analyze prediction results"""
    try:
        confidence = float(prediction_array[0][0])
        
        # Determine result based on model output
        if confidence > 0.5:
            result = "Pneumonia"
            confidence_percent = confidence * 100
        else:
            result = "Normal"
            confidence_percent = (1 - confidence) * 100
        
        analysis = {
            "prediction_result": result,
            "confidence_score": confidence_percent,
            "raw_prediction": confidence,
            "model_threshold": 0.5
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction analysis failed")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Pneumonia Detection API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    try:
        db = get_db()
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "database_status": db_status,
        "timestamp": datetime.utcnow()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_pneumonia(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form("Anonymous"),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None),
    sender_email: Optional[str] = Form(None),
    sender_password: Optional[str] = Form(None),
    recipient_email: Optional[str] = Form(None),
    send_notification: Optional[bool] = Form(False),
    db: PneumoniaDatabase = Depends(get_db)
):
    """Predict pneumonia from chest X-ray image with optional email notification"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Analyze results
        analysis = analyze_prediction(prediction)
        
        # Generate clinical notes (simplified)
        clinical_notes = [
            f"AI Analysis Result: {analysis['prediction_result']}",
            f"Confidence Level: {analysis['confidence_score']:.1f}%",
            "Note: This AI analysis is for screening purposes only.",
            "Professional medical interpretation is required for final diagnosis."
        ]
        
        # Save to database
        prediction_data = {
            "user_id": user_id,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "prediction_result": analysis["prediction_result"],
            "confidence_score": analysis["confidence_score"],
            "image_filename": file.filename,
            "image_size": len(image_bytes),
            "analysis_data": analysis,
            "clinical_notes": clinical_notes
        }
        
        prediction_id = db.save_prediction(prediction_data)
        
        # Save image to GridFS
        image_metadata = {
            "prediction_id": prediction_id,
            "patient_name": patient_name,
            "content_type": file.content_type
        }
        db.save_image(image_bytes, file.filename, image_metadata)
        
        # Send email notification if requested
        email_result = None
        if send_notification and sender_email and recipient_email:
            try:
                from email_utils import email_sender
                
                # Prepare prediction data for email
                email_prediction_data = {
                    'result': analysis["prediction_result"],
                    'confidence': analysis["confidence_score"],
                    'patient_name': patient_name
                }
                
                # Generate email content
                subject, body = email_sender.generate_email_content(email_prediction_data)
                
                # Send email
                email_result = email_sender.send_email(
                    sender_email=sender_email,
                    recipient=recipient_email,
                    subject=subject,
                    body=body,
                    sender_password=sender_password
                )
                
                # Log email result but don't fail the prediction
                if email_result['success']:
                    logger.info(f"Email notification sent successfully to {recipient_email}")
                else:
                    logger.warning(f"Email notification failed: {email_result['error']}")
                    
            except Exception as e:
                logger.error(f"Email notification error: {str(e)}")
                email_result = {
                    "success": False,
                    "error": f"Email notification failed: {str(e)}"
                }
        
        response = PredictionResponse(
            id=prediction_id,
            prediction_result=analysis["prediction_result"],
            confidence_score=analysis["confidence_score"],
            patient_name=patient_name,
            timestamp=datetime.utcnow(),
            analysis_data=analysis,
            clinical_notes=clinical_notes
        )
        
        # Add email result to response if email was attempted
        if email_result:
            response.email_notification = email_result
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/send-notification", response_model=EmailNotificationResponse)
async def send_email_notification(
    notification_request: EmailNotificationRequest,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Send email notification for a specific prediction"""
    
    try:
        # Get prediction from database
        prediction = db.get_prediction_by_id(notification_request.prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        from email_utils import email_sender
        
        # Prepare prediction data for email
        email_prediction_data = {
            'result': prediction["prediction_result"],
            'confidence': prediction["confidence_score"],
            'patient_name': prediction.get("patient_name", "Anonymous")
        }
        
        # Generate email content
        subject, body = email_sender.generate_email_content(email_prediction_data)
        
        # Send email
        result = email_sender.send_email(
            sender_email=notification_request.sender_email,
            recipient=notification_request.recipient_email,
            subject=subject,
            body=body,
            sender_password=notification_request.sender_password
        )
        
        return EmailNotificationResponse(
            success=result['success'],
            message=result.get('message', 'Email sent successfully' if result['success'] else 'Email failed'),
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email notification error: {str(e)}")
        return EmailNotificationResponse(
            success=False,
            message="Email notification failed",
            error=str(e)
        )

@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(
    user_id: Optional[str] = None,
    limit: int = 50,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Get predictions from database"""
    try:
        predictions = db.get_predictions(user_id=user_id, limit=limit)
        
        response_data = []
        for pred in predictions:
            response_data.append(PredictionResponse(
                id=pred["_id"],
                prediction_result=pred["prediction_result"],
                confidence_score=pred["confidence_score"],
                patient_name=pred.get("patient_name", "Anonymous"),
                timestamp=pred["timestamp"],
                analysis_data=pred.get("analysis_data"),
                clinical_notes=pred.get("clinical_notes")
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve predictions")

@app.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Get specific prediction by ID"""
    try:
        prediction = db.get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve prediction")

@app.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Create new user"""
    try:
        user_dict = user_data.dict()
        user_id = db.save_user(user_dict)
        
        # Retrieve the created user
        created_user = db.get_user_by_email(user_data.email)
        
        return UserResponse(
            id=created_user["_id"],
            name=created_user["name"],
            email=created_user["email"],
            role=created_user["role"],
            department=created_user.get("department"),
            hospital=created_user.get("hospital"),
            created_at=created_user["created_at"],
            is_active=created_user["is_active"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.get("/users/{email}")
async def get_user(
    email: str,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Get user by email"""
    try:
        user = db.get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")

@app.post("/reports")
async def create_medical_report(
    report_data: MedicalReportCreate,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Create medical report"""
    try:
        report_dict = report_data.dict()
        report_id = db.save_medical_report(report_dict)
        
        return {"id": report_id, "message": "Medical report created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating medical report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create medical report")

@app.get("/statistics")
async def get_statistics(db: PneumoniaDatabase = Depends(get_db)):
    """Get database statistics"""
    try:
        stats = db.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.put("/predictions/{prediction_id}/status")
async def update_prediction_status(
    prediction_id: str,
    status: str,
    notes: Optional[str] = None,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Update prediction status"""
    try:
        success = db.update_prediction_status(prediction_id, status, notes)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"message": "Prediction status updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prediction status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update prediction status")

@app.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    db: PneumoniaDatabase = Depends(get_db)
):
    """Delete prediction"""
    try:
        success = db.delete_prediction(prediction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"message": "Prediction deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete prediction")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
