import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from matplotlib.widgets import Button
import tkinter as tk

class ImageVisualizer:
    def __init__(self):
        self.current_group = 0
        self.groups = []
        self.fig = None
        self.axes = None

    def get_layout(self, n_images):
        """Determine layout based on number of images"""
        if n_images <= 2:
            return 1, n_images  # Use actual number of columns, maximize images
        elif n_images <= 4:
            return 1, n_images  # Use actual number of columns
        elif n_images <= 6:
            return 2, 3  # 2 rows, 3 columns, maximize images
        else:
            return 3, 3  # 3 rows, 3 columns, maximize images

    def show_all_groups(self, similar_groups):
        """Display all similar image groups, support left and right arrow navigation"""
        self.groups = [sorted(group, key=lambda x: x[1], reverse=True) 
                      for group in similar_groups]
        self.current_group = 0
        
        # Get screen resolution
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Get maximum image count and calculate layout
        max_images = max(len(group) for group in self.groups)
        rows, cols = self.get_layout(max_images)
        
        # Set image display size
        dpi = 96
        fig_width = screen_width / dpi * 0.95
        
        # Adjust height based on number of rows and images
        if rows == 1:
            fig_height = screen_height / dpi * 0.85  # Increase single row height
        else:
            fig_height = screen_height / dpi * 0.9  # Increase multi-row height
        
        # Create figure window
        plt.rcParams['figure.figsize'] = [fig_width, fig_height]
        plt.rcParams['figure.dpi'] = dpi
        
        self.fig, self.axes = plt.subplots(rows, cols)
        self.axes = np.atleast_2d(self.axes)  # Ensure 2D array
        
        # Configure keyboard and toolbar navigation
        def handle_navigation(event):
            if isinstance(event, str):
                direction = event
            else:
                direction = event.key
            
            if direction in ['right', 'forward'] and self.current_group < len(self.groups) - 1:
                self.current_group += 1
                self.update_display()
            elif direction in ['left', 'back'] and self.current_group > 0:
                self.current_group -= 1
                self.update_display()
        
            # Add keyboard event listener
        self.fig.canvas.mpl_connect('key_press_event', handle_navigation)
        
        # Configure toolbar buttons
        toolbar = self.fig.canvas.manager.toolbar
        if hasattr(toolbar, '_actions'):  # PyQt backend
            for action in toolbar._actions.values():
                if action.text() == 'Forward':
                    action.triggered.connect(lambda: handle_navigation('forward'))
                elif action.text() == 'Back':
                    action.triggered.connect(lambda: handle_navigation('back'))
        elif hasattr(toolbar, 'actions'):  # Qt5 backend
            try:
                forward_action = toolbar.actions()[7]
                back_action = toolbar.actions()[6]
                forward_action.triggered.connect(lambda: handle_navigation('forward'))
                back_action.triggered.connect(lambda: handle_navigation('back'))
            except:
                print("Warning: Could not configure toolbar buttons")
        
        # Initial display
        self.update_display()
        plt.show()

    def update_display(self):
        """Update current displayed image group"""
        if not self.groups:
            return
            
        group = self.groups[self.current_group]
        n_images = len(group)
        
        # Update window title
        self.fig.canvas.manager.set_window_title(
            f'Similar Images - Group {self.current_group + 1}/{len(self.groups)}'
        )
        
        # Adjust layout based on number of rows and images
        rows, cols = self.get_layout(n_images)
        if rows == 1:
            # Single row layout
            plt.subplots_adjust(
                left=0.05, right=0.95,  # Increase margin
                top=0.85,  # Reduce top space to avoid toolbar遮挡
                bottom=0.1,  # Increase bottom space
                wspace=0.1,  # Reduce image spacing
                hspace=0.35
            )
        elif rows == 2:
            # Two row layout
            plt.subplots_adjust(
                left=0.05, right=0.95,
                top=0.88,  # Reduce top space to avoid toolbar遮挡
                bottom=0.1,
                wspace=0.1,
                hspace=0.5  # Increase row spacing
            )
        else:
            # Three row layout
            plt.subplots_adjust(
                left=0.05, right=0.95,
                top=0.9,
                bottom=0.1,
                wspace=0.1,
                hspace=0.4
            )
        
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
                ax.axis('off')
                ax.set_visible(False)
        
        # Display images in current group
        for idx, (path, score, metrics) in enumerate(group):
            row = idx // cols
            col = idx % cols
            
            if row < self.axes.shape[0] and col < self.axes.shape[1]:
                self.axes[row, col].set_visible(True)
                # Read image
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Add score to image
                height, width = img.shape[:2]
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = min(width, height) / 1000
                thickness = max(2, int(font_scale * 2))
                
                # Create semi-transparent background
                overlay = img.copy()
                bg_height = int(250 * font_scale)  # Increase background height
                bg_width = int(350 * font_scale)   # Increase background width
                cv2.rectangle(overlay, (0, 0), (bg_width, bg_height), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                
                # Add score information
                y_offset = int(bg_height * 0.15)
                line_height = int(bg_height * 0.18)  # Increase line spacing
                
                # Total score
                self._add_text_with_outline(img, f"Total: {score:.1f}", 10, y_offset, 
                                          font, font_scale, thickness)
                
                # Sub-score
                y_offset += line_height
                norm_scores = metrics['normalized_scores']
                self._add_text_with_outline(img, 
                    f"Tech: {norm_scores['resolution']:.1f} ({metrics['resolution']/1_000_000:.1f}MP)", 
                    10, y_offset, font, font_scale*0.8, thickness)
                
                y_offset += line_height
                self._add_text_with_outline(img,
                    f"Clarity: {norm_scores['clarity']:.1f} ({metrics['clarity']:.0f})", 
                    10, y_offset, font, font_scale*0.8, thickness)
                
                y_offset += line_height
                self._add_text_with_outline(img,
                    f"Face: {norm_scores['face_score']:.1f}",
                    10, y_offset, font, font_scale*0.8, thickness)
                
                y_offset += line_height
                self._add_text_with_outline(img,
                    f"Aesth: {norm_scores['aesthetic']:.1f}",
                    10, y_offset, font, font_scale*0.8, thickness)
                
                # Display image
                self.axes[row, col].imshow(img)
                
                # Update title, only show group number, index, and file name
                title = f"Group {self.current_group + 1} - #{idx + 1}\n{os.path.basename(path)}"
                
                if rows == 1:
                    pad = 20
                    fontsize = 11
                else:
                    pad = 30
                    fontsize = 10
                
                self.axes[row, col].set_title(
                    title,
                    pad=pad,
                    fontsize=fontsize,
                    linespacing=1.5
                )
        
        self.fig.canvas.draw_idle()

    def _add_text_with_outline(self, img, text, x, y, font, font_scale, thickness):
        """Add text with outline"""
        # Add black outline
        cv2.putText(img, text, (x, y), font, font_scale,
                   (0, 0, 0), thickness + 1)
        # Add white text
        cv2.putText(img, text, (x, y), font, font_scale,
                   (255, 255, 255), thickness)

    @staticmethod
    def draw_score_on_image(image_path, score):
        """Draw score on image and save"""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Set font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 1000
        thickness = max(1, int(font_scale * 2))
        
        # Create semi-transparent background
        overlay = img.copy()
        pt1 = (0, 0)
        pt2 = (200, 40)
        cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), -1)
        alpha = 0.6
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Add score text
        score_text = f"Score: {score:.1f}"
        cv2.putText(img, score_text, (10, 30), font, font_scale,
                   (255, 255, 255), thickness)
        
        # Save or return image
        output_path = f"{os.path.splitext(image_path)[0]}_scored{os.path.splitext(image_path)[1]}"
        cv2.imwrite(output_path, img)
        return output_path 