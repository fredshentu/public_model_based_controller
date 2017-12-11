import datetime
import dateutil.tz

def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

def exp_name(algo, env_name, **kwargs):
	exp_name = algo + '_' + env_name
	return exp_name + '_' + '_'.join([str(k) + str(v) for k, v in kwargs.items()])