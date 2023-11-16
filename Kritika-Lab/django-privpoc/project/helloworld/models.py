from django.db import models
from datetime import datetime 

# Create your models here.
class Latest_DateTime(models.Model):
    timestamp = models.DateTimeField()
    rando = models.DecimalField(max_digits=12, decimal_places=0)

    def save(self, *args, **kwargs):
        if not self.id:
            self.timestamp = datetime.utcnow()
        return super(Latest_DateTime, self).save(*args, **kwargs)