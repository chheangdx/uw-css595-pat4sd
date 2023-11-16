from django.apps import AppConfig


class HelloworldConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'helloworld'

    #run the scheduler when the app starts
    def ready(self):
        from scheduler import updater
        updater.start()

