
import pymongo
from pymongo import MongoClient
from datetime import datetime
import os
from typing import Dict, List, Optional
import logging
from bson import ObjectId
import gridfs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PneumoniaDatabase:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "pneum_project"):
        """Initialize MongoDB connection for pneumonia project"""
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[db_name]
            self.fs = gridfs.GridFS(self.db)
            
            # Collections
            self.predictions = self.db.predictions
            self.users = self.db.users
            self.medical_reports = self.db.medical_reports
            self.model_metrics = self.db.model_metrics
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database: {db_name}")
            
            # Create indexes for better performance
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Predictions collection indexes
            self.predictions.create_index([("timestamp", -1)])
            self.predictions.create_index([("user_id", 1)])
            self.predictions.create_index([("prediction_result", 1)])
            
            # Users collection indexes
            self.users.create_index([("email", 1)], unique=True)
            self.users.create_index([("created_at", -1)])
            
            # Medical reports collection indexes
            self.medical_reports.create_index([("prediction_id", 1)])
            self.medical_reports.create_index([("created_at", -1)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes: {str(e)}")
    
    def save_prediction(self, prediction_data: Dict) -> str:
        """Save pneumonia prediction to database"""
        try:
            prediction_doc = {
                "user_id": prediction_data.get("user_id"),
                "patient_name": prediction_data.get("patient_name", "Anonymous"),
                "patient_age": prediction_data.get("patient_age"),
                "patient_gender": prediction_data.get("patient_gender"),
                "prediction_result": prediction_data["prediction_result"],
                "confidence_score": prediction_data["confidence_score"],
                "image_filename": prediction_data.get("image_filename"),
                "image_size": prediction_data.get("image_size"),
                "preprocessing_method": prediction_data.get("preprocessing_method", "colab_style"),
                "model_version": prediction_data.get("model_version", "pneumonia_model.keras"),
                "analysis_data": prediction_data.get("analysis_data"),
                "clinical_notes": prediction_data.get("clinical_notes"),
                "timestamp": datetime.utcnow(),
                "status": "completed"
            }
            
            result = self.predictions.insert_one(prediction_doc)
            logger.info(f"Prediction saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise
    
    def get_predictions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Retrieve predictions from database"""
        try:
            query = {}
            if user_id:
                query["user_id"] = user_id
            
            predictions = list(
                self.predictions.find(query)
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string for JSON serialization
            for pred in predictions:
                pred["_id"] = str(pred["_id"])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict]:
        """Get specific prediction by ID"""
        try:
            prediction = self.predictions.find_one({"_id": ObjectId(prediction_id)})
            if prediction:
                prediction["_id"] = str(prediction["_id"])
            return prediction
        except Exception as e:
            logger.error(f"Error retrieving prediction {prediction_id}: {str(e)}")
            return None
    
    def save_user(self, user_data: Dict) -> str:
        """Save user information"""
        try:
            user_doc = {
                "name": user_data["name"],
                "email": user_data["email"],
                "role": user_data.get("role", "user"),
                "department": user_data.get("department"),
                "hospital": user_data.get("hospital"),
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "is_active": True
            }
            
            result = self.users.insert_one(user_doc)
            logger.info(f"User saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except pymongo.errors.DuplicateKeyError:
            logger.warning(f"User with email {user_data['email']} already exists")
            raise ValueError("User with this email already exists")
        except Exception as e:
            logger.error(f"Error saving user: {str(e)}")
            raise
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            user = self.users.find_one({"email": email})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error retrieving user {email}: {str(e)}")
            return None
    
    def save_medical_report(self, report_data: Dict) -> str:
        """Save medical report to database"""
        try:
            report_doc = {
                "prediction_id": report_data["prediction_id"],
                "patient_info": report_data.get("patient_info", {}),
                "findings": report_data["findings"],
                "clinical_notes": report_data["clinical_notes"],
                "radiologist_notes": report_data.get("radiologist_notes"),
                "report_type": report_data.get("report_type", "ai_screening"),
                "status": report_data.get("status", "draft"),
                "created_at": datetime.utcnow(),
                "created_by": report_data.get("created_by")
            }
            
            result = self.medical_reports.insert_one(report_doc)
            logger.info(f"Medical report saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving medical report: {str(e)}")
            raise
    
    def save_image(self, image_data: bytes, filename: str, metadata: Dict = None) -> str:
        """Save image to GridFS"""
        try:
            file_id = self.fs.put(
                image_data,
                filename=filename,
                metadata=metadata or {},
                upload_date=datetime.utcnow()
            )
            logger.info(f"Image saved with ID: {file_id}")
            return str(file_id)
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise
    
    def get_image(self, file_id: str) -> Optional[bytes]:
        """Retrieve image from GridFS"""
        try:
            grid_out = self.fs.get(ObjectId(file_id))
            return grid_out.read()
        except Exception as e:
            logger.error(f"Error retrieving image {file_id}: {str(e)}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {
                "total_predictions": self.predictions.count_documents({}),
                "pneumonia_cases": self.predictions.count_documents({"prediction_result": "Pneumonia"}),
                "normal_cases": self.predictions.count_documents({"prediction_result": "Normal"}),
                "total_users": self.users.count_documents({}),
                "total_reports": self.medical_reports.count_documents({}),
                "recent_predictions": self.predictions.count_documents({
                    "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
                })
            }
            
            # Calculate accuracy if we have ground truth data
            total_cases = stats["pneumonia_cases"] + stats["normal_cases"]
            if total_cases > 0:
                stats["pneumonia_rate"] = (stats["pneumonia_cases"] / total_cases) * 100
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def update_prediction_status(self, prediction_id: str, status: str, notes: str = None) -> bool:
        """Update prediction status"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            if notes:
                update_data["status_notes"] = notes
            
            result = self.predictions.update_one(
                {"_id": ObjectId(prediction_id)},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating prediction status: {str(e)}")
            return False
    
    def delete_prediction(self, prediction_id: str) -> bool:
        """Delete prediction from database"""
        try:
            result = self.predictions.delete_one({"_id": ObjectId(prediction_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting prediction: {str(e)}")
            return False
    
    def close_connection(self):
        """Close database connection"""
        try:
            self.client.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")

# Global database instance
db_instance = None

def get_database() -> PneumoniaDatabase:
    """Get database instance (singleton pattern)"""
    global db_instance
    if db_instance is None:
        db_instance = PneumoniaDatabase()
    return db_instance

def init_database(connection_string: str = "mongodb://localhost:27017/", db_name: str = "pneum_project"):
    """Initialize database with custom parameters"""
    global db_instance
    db_instance = PneumoniaDatabase(connection_string, db_name)
    return db_instance
