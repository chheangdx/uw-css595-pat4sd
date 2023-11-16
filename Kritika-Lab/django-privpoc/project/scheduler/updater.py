from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from scheduler import updaterApi

def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(updaterApi.update_datetime, 'interval', seconds=10)
    scheduler.start()
