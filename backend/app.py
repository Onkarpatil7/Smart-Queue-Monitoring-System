from fastapi import FastAPI, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from database import sessionLocal
from models import queueData, AlertStatus, ContactMessage
from fastapi.responses import JSONResponse #to return json response
from typing import Optional
import asyncio
from fastapi.middleware.cors import CORSMiddleware


# Simple in-memory connection manager for WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        # send to all connected clients, remove dead connections
        to_remove = []
        for ws in list(self.active_connections):
            try:
                await ws.send_json(message)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(ws)

manager = ConnectionManager()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionData(BaseModel):
    id: int     
    entryTime: datetime      
    exitTime: Optional[datetime] = None
    waitTime: float
    alert: Optional[int] = 0  # 0 = no alert, 1 = alert popped


class ContactMessageRequest(BaseModel):
    name: str
    email: str
    message: str


def get_db():
    db = sessionLocal()  # Create new session
    try:
        yield db
    finally:
        db.close()




@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming detection stats to connected dashboards."""
    await manager.connect(websocket)
    try:
        while True:
            # keep the connection alive; client may send pings but we don't expect messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.post("/contact/")
def save_contact_message(data: ContactMessageRequest, db: Session = Depends(get_db)):
    """
    This route receives contact form data (name, email, message) and saves it to the database.
    """
    try:
        contact_entry = ContactMessage(
            name=data.name,
            email=data.email,
            message=data.message
        )
        
        db.add(contact_entry)
        db.commit()
        db.refresh(contact_entry)
        
        return {
            "message": "Message saved successfully",
            "id": contact_entry.id,
            "name": contact_entry.name,
            "email": contact_entry.email,
            "created_at": contact_entry.created_at.isoformat() if contact_entry.created_at else None
        }
    except Exception as e:
        db.rollback()
        return JSONResponse({"detail": f"Error saving message: {str(e)}"}, status_code=500)


@app.post("/publish/")
async def publish(request: Request):
    """Accepts JSON payloads from `detection.py` and broadcasts to all WebSocket clients.

    Expected: a JSON object containing detection stats (entered, exited, inside, etc.).
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON"}, status_code=400)

    # Broadcast asynchronously (don't block the HTTP response on client slowdowns)
    asyncio.create_task(manager.broadcast(payload))

    return {"message": "published"}

@app.get("/")
def root():
    return {"message":"backend is running"}

@app.post("/updateData/")
def update_detection(data: DetectionData, db: Session = Depends(get_db)):
    """
    This route receives the detected person's ID and waitTime from YOLO script
    and saves them into the PostgreSQL database with a timestamp.
    """
    # Convert alert integer to AlertStatus enum
    alert_status = AlertStatus.ALERT_POPPED if data.alert == 1 else AlertStatus.NO_ALERT
    
    entry = queueData(
        id=data.id,                              # Use YOLO's person ID
        entryTime=data.entryTime,                # Entry timestamp
        exitTime=data.exitTime,                  # Exit timestamp (can be None)
        waitTime=data.waitTime,                  # Save the person's waiting time
        alert=alert_status                       # Alert status (0 or 1)
    )
    
    db.add(entry)
    db.commit()
    db.refresh(entry)  # Refresh to get any auto-generated fields

    return {
        "message": "Data saved successfully",
        "id": entry.id,
        "entryTime": entry.entryTime.isoformat() if entry.entryTime else None,
        "exitTime": entry.exitTime.isoformat() if entry.exitTime else None,
        "waitTime": entry.waitTime,
        "alert": entry.alert.value if entry.alert else 0
    }

#to get total count of rows in database
@app.get("/total-count")
def total_count():
    db = sessionLocal()
    try:
        count = db.query(queueData).count()
        return {"total_count": count}
    finally:
        db.close()

#to get average wait time from all records
@app.get("/avg-waittime")
def avg_waittime():
    db = sessionLocal()
    try:
        rows = db.query(queueData).filter(queueData.waitTime != None).all()
        if not rows:
            return {"average_wait_time": 0}
        # Only calculate average from non-null waitTime values
        total_wait = sum(r.waitTime for r in rows if r.waitTime is not None)
        avg_wait = total_wait / len(rows) if rows else 0
        return {"average_wait_time": float(avg_wait)}
    finally:
        db.close()

#to get last 10 entries from database
@app.get("/recent-entries")
def recent_entries():
    db = sessionLocal()
    try:
        rows = (
            db.query(queueData)
            .order_by(queueData.exitTime.desc())
            .limit(10)
            .all()
        )

        return [
            {
                "id": r.id,
                "entryTime": r.entryTime,
                "exitTime": r.exitTime,
                "waitTime": r.waitTime,
                "alert": r.alert
            }
            for r in rows
        ]
    finally:
        db.close()


