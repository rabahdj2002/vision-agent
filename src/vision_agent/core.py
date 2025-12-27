import time
import cv2
import numpy as np
import pyautogui
import easyocr
import mss
from PIL import Image

class VisionAgent:
    def __init__(self, use_gpu=False, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("👁️ VisionAgent: Loading AI models (this may take a moment)...")
        
        # Suppress EasyOCR warning if GPU is missing
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        self.sct = mss.mss()
        
        # Safety: Fail-safe if mouse hits corner
        pyautogui.FAILSAFE = True 

    def _capture_screen(self):
        """Captures the screen efficiently."""
        monitor = self.sct.monitors[1]
        screenshot = np.array(self.sct.grab(monitor))
        # Remove Alpha channel (BGRA -> BGR)
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    def log(self, message):
        if self.verbose:
            print(f"[VisionAgent] {message}")

    def click_text(self, text, double_click=False, confidence=0.5):
        """Finds text on screen and clicks it."""
        self.log(f"🔎 Scanning for text: '{text}'...")
        
        img = self._capture_screen()
        results = self.reader.readtext(img)

        for (bbox, detected_text, prob) in results:
            if text.lower() in detected_text.lower() and prob >= confidence:
                self.log(f"✅ Found '{detected_text}' ({prob:.2f})")
                
                (tl, tr, br, bl) = bbox
                center_x = int((tl[0] + br[0]) / 2)
                center_y = int((tl[1] + br[1]) / 2)
                
                # Correction for retina displays (Mac) or scaling might be needed here
                # For now, we assume 1:1 pixel mapping
                
                self._move_and_click(center_x, center_y, double_click)
                return True
        
        self.log(f"❌ Text '{text}' not found.")
        return False

    def click_icon(self, icon_path, confidence=0.8):
        """Finds an icon/image on screen and clicks it."""
        self.log(f"🔎 Scanning for icon: '{icon_path}'...")
        
        screen_img = self._capture_screen()
        
        # Read template
        template = cv2.imread(icon_path, cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(f"Icon file not found: {icon_path}")

        result = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= confidence:
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            
            self.log(f"✅ Found icon (Match: {max_val:.2f})")
            self._move_and_click(center_x, center_y)
            return True
        
        self.log(f"❌ Icon not found (Best match: {max_val:.2f})")
        return False

    def _move_and_click(self, x, y, double=False):
        pyautogui.moveTo(x, y, duration=0.5)
        if double:
            pyautogui.doubleClick()
        else:
            pyautogui.click()

    def type(self, text):
        self.log(f"⌨️ Typing: {text}")
        pyautogui.write(text, interval=0.05)