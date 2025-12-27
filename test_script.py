from vision_agent import VisionAgent
import time

bot = VisionAgent(verbose=True)

print("--- DEMO STARTING ---")
time.sleep(0)

# Scan only the top-left 500x500 pixels
# Format: (left, top, width, height)
bot.click_text("rabahdj2002", debug=True)
time.sleep(3)
bot.click_text("vision-agent-project", debug=True)
