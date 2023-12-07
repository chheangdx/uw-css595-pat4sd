from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from helloworld.models import Latest_DateTime

# Create your views here.
def homePageView(request):
    return HttpResponse("Hello, World!")

def schedulerDemoView(request):
    latest_datetime = Latest_DateTime.objects.latest('timestamp')
    rando = latest_datetime.rando
    timestamp = "{t.year}/{t.month:02d}/{t.day:02d} - {t.hour:02d}:{t.minute:02d}:{t.second:02d}".format( t=latest_datetime.timestamp)

    context = {
        "datetime_updated": timestamp,
        "rando": rando,
    }
    return render(request, "scheduler_demo.html", context)

