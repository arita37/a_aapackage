#####################################################################################
############################ Celery Code ############################################
from celery import Celery

app = Celery('tasks_folder', backend='amqp', broker='amqp://localhost//')
# app.config_from_object('config')
app.conf.update( BROKER_HEARTBEAT = 10,
    CELERY_ACKS_LATE = True,
    CELERYD_PREFETCH_MULTIPLIER = 1,
    CELERY_TRACK_STARTED = True, )


'''
app = Celery('tasks_folder', backend='amqp', broker='amqp://guest@my.server.com//')

app.conf.update( BROKER_HEARTBEAT = 10,
    CELERY_ACKS_LATE = True,
    CELERYD_PREFETCH_MULTIPLIER = 1,
    CELERY_TRACK_STARTED = True, )
    CELERY_RESULT_BACKEND=
    CELERY_RESULT_DBURI=

'''

@app.task(trail=True)
def task01(how_many):
    return 15



@app.task(trail=True)
def A(how_many):
    return group(B.s(i) for i in range(how_many))()



@app.task(trail=True)
def B(i):
    return pow2.delay(i)



@app.task(trail=True)
def pow2(i):
    return i ** 2






