class FeedbackLearner:
    def __init__(self):
        self.feedback_history = []
        
    def learn_from_feedback(self, selected_image, rejected_images, metrics):
        """从用户选择中学习调整权重"""
        self.feedback_history.append({
            'selected': selected_image,
            'rejected': rejected_images,
            'metrics': metrics
        })
        
        # 更新权重模型
        self.update_weights() 