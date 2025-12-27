from vision_agent import VisionAgent
import time

# 1. Start the bot
bot = VisionAgent(verbose=True)

print("--- DEMO STARTING IN 3 SECONDS ---")
print("Please open Calculator or Notepad so I have something to click!")
time.sleep(3)

# 2. Try to find and click the Windows Start button
# (This works on Windows 10/11 because the start button literally says "Start" in the code, or usually has a tooltip)
# If that fails, let's try a common desktop app text like "Recycle" or "File"
bot.click_text("clicked", double_click=True)

# 3. If you have Notepad open, try this:
# bot.click_text("File")
# bot.type("Hello world this is vision agent")