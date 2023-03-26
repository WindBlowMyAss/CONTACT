import multiprocessing

bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1

capture_output = True

accesslog = "-"
access_log_format = "[%(t)s]\t %(h)s %(r)s"
errorlog = '-'
