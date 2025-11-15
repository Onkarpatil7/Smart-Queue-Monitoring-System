# AI Queue Detector

An intelligent queue management system that uses YOLOv8 for real-time person detection and tracking. The system monitors entry/exit events, calculates wait times, and provides crowd alerts when the maximum capacity is reached. All data is stored in a PostgreSQL database and can be viewed through a web dashboard.

## Features

- 🎯 **Real-time Person Detection**: Uses YOLOv8 model for accurate person detection
- 📊 **Queue Tracking**: Tracks entry and exit times for each person
- ⏱️ **Wait Time Calculation**: Automatically calculates wait time based on entry and exit timestamps
- 🚨 **Crowd Alert System**: Alerts when the maximum capacity limit is reached
- 💾 **Database Storage**: Stores all detection data in PostgreSQL database
- 📈 **Web Dashboard**: Real-time dashboard to view queue statistics
- 🔔 **Alert Status Tracking**: Records alert status (0 or 1) for each detection event

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL
- **AI/ML**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Frontend**: HTML, JavaScript
- **ORM**: SQLAlchemy

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- PostgreSQL database server
- Webcam or camera device (for detection)
- pip (Python package manager)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "AI Queue Demo"
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Navigate to the `backend` directory and install the required packages:

```bash
cd backend
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL Database

1. **Install PostgreSQL** (if not already installed)
   - Download from [PostgreSQL Official Website](https://www.postgresql.org/download/)

2. **Create a Database**
   ```sql
   CREATE DATABASE queue_detector;
   ```

### 5. Configure Environment Variables

Create a `.env` file in the `backend` directory with your PostgreSQL database configuration:

```env
DB_HOST=localhost
DB_NAME=queue_detector
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432
```

**Example `.env` file:**
```env
DB_HOST=localhost
DB_NAME=queue_detector
DB_USER=postgres
DB_PASSWORD=mypassword
DB_PORT=5432
```

### 6. Create Database Tables

Run the `create.py` script **once** to create the necessary database tables:

```bash
cd backend
python create.py
```

This will create the `queuedata` table with the following schema:
- `id` (Integer, Primary Key)
- `entryTime` (DateTime, Primary Key)
- `exitTime` (DateTime)
- `waitTime` (Float)
- `alert` (Enum: 0 = No Alert, 1 = Alert Popped)

## Running the Application

### Step 1: Start the Backend Server

Open a terminal and navigate to the `backend` directory:

```bash
cd backend
uvicorn app:app --reload
```

The FastAPI server will start on `http://localhost:8000`

You can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Step 2: Start the Detection Script

Open a **new terminal** and navigate to the `backend` directory:

```bash
cd backend
python detection.py
```

This will:
- Start the camera feed
- Begin detecting and tracking people
- Send exit data to the database automatically
- Display real-time statistics

**Note**: Press `Q` to quit the detection script.

### Step 3: Open the Frontend Dashboard

Open the `frontend/index.html` file in your web browser:

```bash
# Simply open the file in your browser
# Or use a simple HTTP server:
cd frontend
python -m http.server 8080
# Then navigate to http://localhost:8080
```

The dashboard will automatically refresh every 5 seconds to show the latest queue data.

## Project Structure

```
AI Queue Demo/
│
├── backend/
│   ├── app.py              # FastAPI backend application
│   ├── models.py           # SQLAlchemy database models
│   ├── database.py         # Database connection configuration
│   ├── create.py           # Script to create database tables
│   ├── detection.py        # YOLOv8 person detection and tracking script
│   ├── requirements.txt    # Python dependencies
│   ├── yolov8n.pt         # YOLOv8 model weights
│   └── .env               # Database configuration (create this)
│
├── frontend/
│   └── index.html         # Web dashboard
│
├── data/                  # Data storage directory
├── docs/                  # Documentation
└── README.md             # This file
```

## API Endpoints

### GET `/`
Returns a simple message confirming the backend is running.

### POST `/updateData/`
Receives person detection data and saves it to the database.

**Request Body:**
```json
{
  "id": 1,
  "entryTime": "2024-01-01T10:00:00",
  "exitTime": "2024-01-01T10:05:30",
  "waitTime": 330.5,
  "alert": 0
}
```

### GET `/getData/`
Retrieves all detection data from the database.

**Response:**
```json
[
  {
    "id": 1,
    "entryTime": "2024-01-01T10:00:00",
    "exitTime": "2024-01-01T10:05:30",
    "waitTime": 330.5,
    "alert": 0
  }
]
```

## Database Schema

The `queuedata` table structure:

| Column     | Type      | Description                          |
|------------|-----------|--------------------------------------|
| id         | Integer   | Person ID (Primary Key)              |
| entryTime  | DateTime  | Entry timestamp (Primary Key)        |
| exitTime   | DateTime  | Exit timestamp (nullable)            |
| waitTime   | Float     | Calculated wait time in seconds      |
| alert      | Enum      | Alert status: 0 (No Alert) or 1 (Alert Popped) |

## Configuration

### Detection Settings

You can modify the following constants in `backend/detection.py`:

- `MAX_PEOPLE`: Maximum number of people to track (default: 10)
- `CONF_THRESHOLD`: Detection confidence threshold (default: 0.6)
- `MIN_BOX_AREA`: Minimum bounding box area (default: 4000)

### API Configuration

The API URL in `detection.py` is set to:
```python
API_URL = "http://localhost:8000/updateData/"
```

Change this if your backend runs on a different host or port.

## Troubleshooting

### Database Connection Issues

- Ensure PostgreSQL is running
- Verify `.env` file has correct credentials
- Check if the database exists: `CREATE DATABASE queue_detector;`

### Camera Not Working

- Ensure your camera is connected and not being used by another application
- Check camera permissions in your system settings
- Try changing the camera index in `detection.py`: `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Import Errors

- Make sure you've activated your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

### Port Already in Use

- Change the port in uvicorn: `uvicorn app:app --reload --port 8001`
- Update the API_URL in `detection.py` accordingly

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Feel free to submit issues and enhancement requests!

