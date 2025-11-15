from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from database import sessionLocal
from models import queueData, AlertStatus
from fastapi.responses import JSONResponse #to return json response
from typing import Optional

app = FastAPI()

class DetectionData(BaseModel):
    id: int     
    entryTime: datetime      
    exitTime: Optional[datetime] = None
    waitTime: float
    alert: Optional[int] = 0  # 0 = no alert, 1 = alert popped   
def get_db():
    db = sessionLocal()  # Create new session
    try:
        yield db
    finally:
        db.close()

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


@app.get("/getData/")
def get_detection_data(db: Session = Depends(get_db)):
    """
    This route retrieves all detection data from the database.
    """
    data_entries = db.query(queueData).all()
    results = [
        {
            "id": entry.id,
            "entryTime": entry.entryTime.isoformat() if entry.entryTime else None,
            "exitTime": entry.exitTime.isoformat() if entry.exitTime else None,
            "waitTime": entry.waitTime,
            "alert": entry.alert.value if entry.alert else 0
        }
        for entry in data_entries
    ]
    return JSONResponse(content=results)