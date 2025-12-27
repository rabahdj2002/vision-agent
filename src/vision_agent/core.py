import time
import cv2
import numpy as np
import pyautogui
import easyocr
import mss
import warnings
from difflib import SequenceMatcher

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

class VisionAgent:
    def __init__(self, use_gpu=False, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("👁️ VisionAgent: Loading AI models...")
        
        # 'lat' model is sometimes better for code/symbols than 'en', but 'en' is standard.
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        self.sct = mss.mss()
        pyautogui.FAILSAFE = True 
        
        self.scale_x, self.scale_y = self._get_dpi_scale()
        if self.verbose:
            print(f"⚖️ DPI Scale: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")

    def _get_dpi_scale(self):
        monitor = self.sct.monitors[1]
        return (monitor["width"] / pyautogui.size()[0], monitor["height"] / pyautogui.size()[1])

    def _capture(self, region=None):
        monitor = self.sct.monitors[1]
        if region:
            monitor = {
                "top": int(region[1] * self.scale_y), 
                "left": int(region[0] * self.scale_x), 
                "width": int(region[2] * self.scale_x), 
                "height": int(region[3] * self.scale_y)
            }
        screenshot = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    def log(self, msg):
        if self.verbose: print(f"[VisionAgent] {msg}")

    def _similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _preprocess(self, img_gray, thicken=False):
        """
        Advanced Preprocessing:
        1. Invert if Dark Mode.
        2. Maximize Contrast (Normalize).
        3. Thicken text (Dilation) if requested.
        """
        # 1. Dark Mode Check & Inversion
        if np.mean(img_gray) < 127:
            img_gray = cv2.bitwise_not(img_gray)

        # 2. Maximize Contrast (Stretch Min/Max to 0-255)
        img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)

        # 3. Thicken Text (Crucial for thin fonts)
        if thicken:
            kernel = np.ones((2,2), np.uint8) # 2x2 pixel kernel
            img_gray = cv2.erode(img_gray, kernel, iterations=1) # Erode darker areas = Thicken black text
        
        return img_gray

    def _find_text_coords(self, text, confidence=0.4, region=None, debug=False):
        self.log(f"🔎 Scanning for: '{text}'...")
        
        img_color = self._capture(region=region)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # --- ATTEMPT 1: Normal + Contrast Boost ---
        processed = self._preprocess(img_gray, thicken=False)
        found = self._scan_image(processed, text, confidence, region, debug, img_color, scale=1.0)
        if found: return found
            
        # --- ATTEMPT 2: 2x Zoom + Thicken Text ---
        self.log(f"   ⚠️ Retrying with 2x Zoom + Thickening...")
        h, w = img_gray.shape
        img_2x = cv2.resize(img_gray, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
        
        # Apply thickening here
        processed_2x = self._preprocess(img_2x, thicken=True)
        
        found = self._scan_image(processed_2x, text, confidence, region, debug, img_color, scale=2.0)
        if found: return found

        # --- ATTEMPT 3: Deep Search (3x + Thresholding) ---
        self.log(f"   ⚠️ Retrying with Deep Search (3x + Threshold)...")
        img_3x = cv2.resize(img_gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        processed_3x = self._preprocess(img_3x, thicken=True) # Thicken first
        
        # Binarize
        img_bin = cv2.adaptiveThreshold(processed_3x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        found = self._scan_image(img_bin, text, confidence, region, debug, img_color, scale=3.0)
        if found: return found

        self.log(f"❌ '{text}' not found.")
        return None

    def _scan_image(self, img_source, target_text, confidence, region, debug, debug_img_color, scale=1.0):
        results = self.reader.readtext(img_source)
        candidates = [] 

        for (bbox, detected_text, prob) in results:
            candidates.append(detected_text)
            
            # IMPROVED FUZZY LOGIC: 
            # If the detected text is long (e.g. "user/repo"), check if "user" is inside it 
            # with a loose match.
            
            is_match = False
            sim_score = self._similarity(target_text, detected_text)
            
            if target_text.lower() in detected_text.lower():
                is_match = True
            elif sim_score > 0.5: # Lowered threshold to 0.5 for very hard text
                is_match = True
            
            if is_match and prob >= confidence:
                self.log(f"✅ Found '{target_text}' (Saw: '{detected_text}', Sim: {sim_score:.2f})")
                
                (tl, tr, br, bl) = bbox
                
                # Smart Target: Center of Box
                # (Complex character math is unreliable with fuzzy matching, so we use center)
                box_left = tl[0] / scale
                box_width = (tr[0] - tl[0]) / scale
                box_top = tl[1] / scale
                box_height = (bl[1] - tl[1]) / scale
                
                target_x = int(box_left + (box_width / 2))
                target_y = int(box_top + (box_height / 2))

                if debug:
                    # Draw on the original image (scaled back)
                    cv2.circle(debug_img_color, (target_x, target_y), 5, (0, 255, 0), -1)
                    cv2.imwrite("debug_view.png", debug_img_color)

                final_x = target_x / self.scale_x
                final_y = target_y / self.scale_y
                
                if region:
                    final_x += region[0]
                    final_y += region[1]
                
                return (final_x, final_y)
        
        if scale == 3.0 and self.verbose:
            print(f"      [Debug] Candidates: {candidates[:5]}...")
        return None

    def click_text(self, text, confidence=0.4, region=None, debug=False, offset_x=0, offset_y=0):
        coords = self._find_text_coords(text, confidence, region, debug)
        if coords:
            tx, ty = coords[0] + offset_x, coords[1] + offset_y
            pyautogui.moveTo(tx, ty, duration=0.2)
            pyautogui.click()
            return True
        return False