# NIRU MVP - Deployment & Demo Guide

This guide ensures you are ready to show your project to judges today/tomorrow.

## 1. Fast Online Demo (Localtunnel)
The easiest way to show your local dashboard to judges via a public URL without complex hosting.

1.  **Start your Flask Server:**
    Open a terminal and run:
    ```bash
    python app.py
    ```
2.  **Expose to Public Link:**
    Open a **second terminal** and run:
    ```bash
    npx localtunnel --port 5000
    ```
3.  **Share the Link:**
    Localtunnel will give you a URL (e.g., `https://cold-mountains-stay.loca.lt`).
    *Note: When you first open the link, it might ask for your public IP for verification. You can find it at [whatsmyip.org](https://whatsmyip.org).*

---

## 2. Professionalism (GitHub)
Judges will want to see your codebase.

1.  **Initialize Git:**
    ```bash
    git init
    git add .
    git commit -m "Initial MVP: Video Intelligence Platform"
    ```
2.  **Create Repository on GitHub:**
    Go to [github.com/new](https://github.com/new), create a repository named `NIRU_MVP`, and follow the instructions to push your code.

---

## 3. The "Safety Net" (Recorded Walkthrough)
Internet fails. Servers crash. **Always have a video of your demo ready.**

1.  Use a screen recorder (like OBS or Windows Game Bar `Win+G`).
2.  Record a 3-5 minute video showing:
    -   The Dashboard loading.
    -   Switching between the 4 Isiolo videos.
    -   Showing the "AI Vision" (Annotated) videos.
    -   Explaining a Safety Flag (if present) and an Insertion Point.

---

## 4. Technical Note: Transcripts
Currently, the results show an error for transcripts because `ffmpeg` is missing on your local machine. 
**For the demo:** Focus on the **Object Detection** and **Safety Analysis** which are working perfectly. If you have time to install FFmpeg, you can re-run the processing.
