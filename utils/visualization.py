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
        """根据图片数量确定布局"""
        if n_images <= 2:
            return 1, n_images  # 使用实际数量的列数，让图片最大化
        elif n_images <= 4:
            return 1, n_images  # 使用实际数量的列数
        elif n_images <= 6:
            return 2, 3  # 2行3列，让图片更大
        else:
            return 3, 3  # 3行3列，让图片更大

    def show_all_groups(self, similar_groups):
        """显示所有相似图片组，支持左右箭头导航"""
        self.groups = [sorted(group, key=lambda x: x[1], reverse=True) 
                      for group in similar_groups]
        self.current_group = 0
        
        # 获取屏幕分辨率
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # 获取最大图片数量并计算布局
        max_images = max(len(group) for group in self.groups)
        rows, cols = self.get_layout(max_images)
        
        # 设置图片显示尺寸
        dpi = 96
        fig_width = screen_width / dpi * 0.95
        
        # 根据行数和图片数量调整高度
        if rows == 1:
            fig_height = screen_height / dpi * 0.85  # 增加单行高度
        else:
            fig_height = screen_height / dpi * 0.9  # 增加多行高度
        
        # 创建图形窗口
        plt.rcParams['figure.figsize'] = [fig_width, fig_height]
        plt.rcParams['figure.dpi'] = dpi
        
        self.fig, self.axes = plt.subplots(rows, cols)
        self.axes = np.atleast_2d(self.axes)  # 确保是2D数组
        
        # 配置键盘和工具栏导航
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
        
        # 添加键盘事件监听
        self.fig.canvas.mpl_connect('key_press_event', handle_navigation)
        
        # 配置工具栏按钮
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
        
        # 初始显示
        self.update_display()
        plt.show()

    def update_display(self):
        """更新当前显示的图片组"""
        if not self.groups:
            return
            
        group = self.groups[self.current_group]
        n_images = len(group)
        
        # 更新窗口标题
        self.fig.canvas.manager.set_window_title(
            f'Similar Images - Group {self.current_group + 1}/{len(self.groups)}'
        )
        
        # 根据行数和图片数量调整布局
        rows, cols = self.get_layout(n_images)
        if rows == 1:
            # 单行布局
            plt.subplots_adjust(
                left=0.05, right=0.95,  # 增加边距
                top=0.85,  # 减少顶部空间避免工具栏遮挡
                bottom=0.1,  # 增加底部空间
                wspace=0.1,  # 减小图片间距
                hspace=0.35
            )
        elif rows == 2:
            # 两行布局
            plt.subplots_adjust(
                left=0.05, right=0.95,
                top=0.88,  # 减少顶部空间避免工具栏遮挡
                bottom=0.1,
                wspace=0.1,
                hspace=0.5  # 增加行间距
            )
        else:
            # 三行布局
            plt.subplots_adjust(
                left=0.05, right=0.95,
                top=0.9,
                bottom=0.1,
                wspace=0.1,
                hspace=0.4
            )
        
        # 清除所有axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
                ax.axis('off')
                ax.set_visible(False)
        
        # 显示当前组的图片
        for idx, (path, score, metrics) in enumerate(group):
            row = idx // cols
            col = idx % cols
            
            if row < self.axes.shape[0] and col < self.axes.shape[1]:
                self.axes[row, col].set_visible(True)
                # 读取图片
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 在图片上添加分数
                height, width = img.shape[:2]
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = min(width, height) / 1000
                thickness = max(2, int(font_scale * 2))
                
                # 创建半透明背景
                overlay = img.copy()
                bg_height = int(250 * font_scale)  # 增加背景高度
                bg_width = int(350 * font_scale)   # 增加背景宽度
                cv2.rectangle(overlay, (0, 0), (bg_width, bg_height), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                
                # 添加分数信息
                y_offset = int(bg_height * 0.15)
                line_height = int(bg_height * 0.18)  # 增加行距
                
                # 总分
                self._add_text_with_outline(img, f"Total: {score:.1f}", 10, y_offset, 
                                          font, font_scale, thickness)
                
                # 分项分数
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
                
                # 显示图片
                self.axes[row, col].imshow(img)
                
                # 更新标题，只显示组号、编号和文件名
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
        """添加带描边的文字"""
        # 添加黑色描边
        cv2.putText(img, text, (x, y), font, font_scale,
                   (0, 0, 0), thickness + 1)
        # 添加白色文字
        cv2.putText(img, text, (x, y), font, font_scale,
                   (255, 255, 255), thickness)

    @staticmethod
    def draw_score_on_image(image_path, score):
        """在图片上绘制分数并保存"""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 1000
        thickness = max(1, int(font_scale * 2))
        
        # 创建半透明背景
        overlay = img.copy()
        pt1 = (0, 0)
        pt2 = (200, 40)
        cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), -1)
        alpha = 0.6
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # 添加分数文本
        score_text = f"Score: {score:.1f}"
        cv2.putText(img, score_text, (10, 30), font, font_scale,
                   (255, 255, 255), thickness)
        
        # 保存或返回图片
        output_path = f"{os.path.splitext(image_path)[0]}_scored{os.path.splitext(image_path)[1]}"
        cv2.imwrite(output_path, img)
        return output_path 