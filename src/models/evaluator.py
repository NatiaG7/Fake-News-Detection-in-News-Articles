from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import json

class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics"""
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def get_accuracy(self) -> float:
        """Calculate accuracy score""" 
        return accuracy_score(self.y_true, self.y_pred)
    
    def get_precision(self, average='weighted') -> float:
        """Calculate precision score""" 
        return precision_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def get_recall(self, average='weighted') -> float:
        """Calculate recall score""" 
        return recall_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def get_f1_score(self, average='weighted') -> float:
        """Calculate F1 score""" 
        return f1_score(self.y_true, self.y_pred, average=average, zero_division=0)
    
    def get_confusion_matrix(self) -> dict:
        """Get confusion matrix""" 
        cm = confusion_matrix(self.y_true, self.y_pred)
        return {
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1]
        }
    
    def get_all_metrics(self) -> dict:
        """Get all evaluation metrics""" 
        return {
            'accuracy': self.get_accuracy(),
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1_score': self.get_f1_score(),
            'confusion_matrix': self.get_confusion_matrix(),
            'classification_report': classification_report(self.y_true, self.y_pred, output_dict=True)
        }
    
    def meets_success_criteria(self, f1_threshold: float = 0.85) -> bool:
        """Check if model meets project success criteria (F1 >= 0.85)""" 
        return self.get_f1_score() >= f1_threshold
