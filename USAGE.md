# AuraBot Usage Guide

## Quick Start

### Check if Container is Running
```bash
docker ps
```

You should see `aurabot-detection` with status `Up`.

### View Live Logs
```bash
docker logs -f aurabot-detection
```

Press `Ctrl+C` to exit log view.

### Stop the Application
```bash
docker-compose down
```

### Start the Application
```bash
docker-compose up -d aurabot-detection
```

---

## Enabling Features

### 1. Camera Support (Object Detection)

Edit `.env` file:
```env
CAMERA_ENABLED=true
```

Restart:
```bash
docker-compose down
docker-compose up -d --build aurabot-detection
```

**Note:** Camera access in Docker on Windows requires special setup (see Windows GUI section below).

### 2. Arduino Support

Connect Arduino via USB, then edit `.env`:
```env
ARDUINO_ENABLED=true
```

Restart:
```bash
docker-compose down
docker-compose up -d --build aurabot-detection
```

### 3. GUI Display (Windows)

**For Windows users, Docker containers cannot display GUI directly.**

You need an X11 server:

#### Option A: Install VcXsrv (Recommended)

1. Download and install VcXsrv from: https://sourceforge.net/projects/vcxsrv/

2. Run XLaunch with these settings:
   - Multiple windows
   - Display number: 0
   - Start no client
   - **UNCHECK** "Native opengl"
   - **CHECK** "Disable access control"

3. Find your Windows IP:
   ```powershell
   ipconfig
   ```
   Look for `IPv4 Address` under your active adapter (e.g., `192.168.1.100`)

4. Edit `docker-compose.yml`:
   ```yaml
   environment:
     - DISPLAY=host.docker.internal:0.0
     - GUI_ENABLED=true
   ```

5. Restart:
   ```bash
   docker-compose down
   docker-compose up -d --build aurabot-detection
   ```

#### Option B: Save Output to File (No GUI needed)

The app saves detected frames. Check the `./temp` folder for output images.

### 4. Face Recognition Module

Start the face recognition service:
```bash
docker-compose --profile faceid up -d aurabot-faceid
```

View logs:
```bash
docker logs -f aurabot-faceid
```

---

## Common Commands

| Command | Description |
|---------|-------------|
| `docker ps` | List running containers |
| `docker logs aurabot-detection` | View application output |
| `docker logs -f aurabot-detection` | View live output (follow) |
| `docker-compose down` | Stop all services |
| `docker-compose up -d` | Start in background |
| `docker exec -it aurabot-detection bash` | Enter container shell |
| `docker images` | List Docker images |
| `docker system prune` | Clean up unused Docker data |

---

## Troubleshooting

### Container Keeps Restarting
```bash
docker logs aurabot-detection
```
Check the error message and ensure your `.env` settings are correct.

### Camera Not Found
- Ensure camera is connected
- Try `CAMERA_ENABLED=false` to run in demo mode
- On Windows, Docker camera access is limited; consider running Python directly without Docker for full camera support

### No GUI Display
- Windows requires X11 server (VcXsrv)
- Make sure `DISPLAY` environment variable is set correctly
- Try disabling GUI: `GUI_ENABLED=false`

### Arduino Not Found
- Check USB connection
- Verify Arduino is recognized in Windows Device Manager
- Try different USB port

---

## Running Without Docker (Alternative)

If Docker limits are causing issues, you can run directly:

```bash
# Install dependencies
pip install -r requirements.txt

# Run main detection
python Automatic_detector.py

# Run face recognition
cd face_id
python main.py
```

---

## Configuration File (.env)

```env
# Display settings
DISPLAY=:0

# API Keys (optional - for AI features)
GEMINI_API_KEY=your_key_here
BLOB_READ_WRITE_TOKEN=your_token_here

# Hardware toggles
CAMERA_ENABLED=false
ARDUINO_ENABLED=false
GUI_ENABLED=false

# Application mode
APP_MODE=detection
```

---

## Project Structure

```
AuraBot/
├── Automatic_detector.py    # Main object detection
├── yolov8n.pt              # AI model
├── requirements.txt        # Python packages
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Service orchestration
├── .env                   # Your settings
├── USAGE.md              # This file
└── face_id/              # Face recognition module
    ├── main.py
    ├── face_engine.py
    └── ...
```

---

## Next Steps

1. **Test basic detection**: `docker-compose up -d aurabot-detection`
2. **View logs**: `docker logs -f aurabot-detection`
3. **Enable camera**: Edit `.env` → `CAMERA_ENABLED=true`
4. **Add GUI** (Windows): Install VcXsrv, update `docker-compose.yml`
5. **Push to GitHub**: Ask project owner to set `OMAR` as default branch

---

**Need help?** Check the full report in `CHANGES_REPORT.html`
