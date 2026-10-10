import os
import sys
import threading
import time
import webbrowser

def main():
    try:
        from streamlit.web import cli as stcli
    except Exception as e:
        print("Could not load Streamlit:", e)
        input("Press Enter to exit...")
        return

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    threading.Thread(target=open_browser, daemon=True).start()
    sys.argv = ["streamlit", "run", app_path, "--server.headless=true", "--browser.gatherUsageStats=false"]
    stcli.main()

if __name__ == "__main__":
    main()
