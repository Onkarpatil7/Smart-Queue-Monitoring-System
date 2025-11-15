from sqlalchemy import Column,Integer,Float,String,DateTime,Enum
from database import Base
from datetime import datetime
import enum


class AlertStatus(enum.Enum):
    NO_ALERT = 0  # alert was not popped at timestamp
    ALERT_POPPED = 1  # alert was popped at timestamp


class queueData(Base):
    __tablename__="queuedata"

    id=Column(Integer,primary_key=True,index=True)
    entryTime=Column(DateTime,default=datetime.now,primary_key=True)
    exitTime=Column(DateTime)
    waitTime=Column(Float)
    alert=Column(Enum(AlertStatus),nullable=True,default=AlertStatus.NO_ALERT)

