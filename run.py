import os
import sys
import threading
import time
import webbrowser

import uvicorn


HOST = "127.0.0.1"
PORT = 8080


# Add the app directory to Python's import path.
# This allows modules inside app/ to import each
# other directly, matching the current project structure.
APP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app"
)

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def open_browser():
    time.sleep(1.5)
    webbrowser.open(
        f"http://{HOST}:{PORT}"
    )


if __name__ == "__main__":

    print("=" * 50)
    print("              NeuroVector")
    print("=" * 50)
    print()
    print("Starting Python vector search and RAG engine...")
    print()
    print(f"Web UI: http://localhost:{PORT}")
    print()
    print("Press CTRL+C to stop the server.")
    print("=" * 50)

    threading.Thread(
        target=open_browser,
        daemon=True
    ).start()

    uvicorn.run(
        "app.api:app",
        host=HOST,
        port=PORT,
        reload=False
    )