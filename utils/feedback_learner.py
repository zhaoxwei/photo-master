class FeedbackLearner:
    def __init__(self):
        self.feedback_history = []
        
    def learn_from_feedback(self, selected_image, rejected_images, metrics):
        """Learn from user feedback to adjust weights"""
        self.feedback_history.append({
            'selected': selected_image,
            'rejected': rejected_images,
            'metrics': metrics
        })
        
        # Update weight model
        self.update_weights() 