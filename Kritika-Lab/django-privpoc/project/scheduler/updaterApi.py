import requests
from helloworld.models import Latest_DateTime
import random
import subprocess

def update_datetime():
    try:
        new_datetime = Latest_DateTime()
        new_datetime.rando = random.randint(0, 10)
        new_datetime.save()

        # Run the other script
        print("Running Script")
        output = subprocess.run(["python", "scheduler/scripting_test.py"], capture_output=True)
        print(output.stdout.decode())
    except:
        pass