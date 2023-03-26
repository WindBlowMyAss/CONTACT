#
# Author: ZeroBeyond
# A simple daemon that automatically starts up,
# listen for requests, and executes the command.
#
import argparse
import os
import signal
import subprocess
from time import gmtime, strftime

import daemon
import lockfile
from flask import Flask, request

# constants
work_dir = "/var/tmp/cmd_daemon"
if not os.path.exists(work_dir):
    os.mkdir(work_dir)

stderr_file = open("/var/tmp/cmd_daemon/stderr.txt", "w+")
stdout_file = open("/var/tmp/cmd_daemon/stdout.txt", "w+")
log_file = open("/var/tmp/cmd_daemon/daemon-log.txt", "w+")


def program_cleanup():
    pass


def reload_program_config():
    pass


context = daemon.DaemonContext(
    working_directory=work_dir,
    umask=0o002,
    pidfile=lockfile.FileLock("/var/tmp/cmd_daemon/lock.pid"),
    stderr=stderr_file,
    stdout=stdout_file,
)

context.signal_map = {
    signal.SIGTERM: program_cleanup,
    signal.SIGHUP: "terminate",
    signal.SIGUSR1: reload_program_config,
}

context.files_preserve = [log_file]


def exec(script):
    subprocess.call(script, shell=True)


def create_app(script):
    # start the main program
    app = Flask(__name__)

    @app.route("/", methods=["POST", "GET"])
    def func():
        log_file.write("[{}] {}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Got Request"))
        if request.method == "POST":
            cmd = request.form["command"]
            if cmd == "exec":
                exec(script)
                log_file.write("[{}] {}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Execute the scripts"))
                return "Execute the scripts!"
            elif cmd == "exit":
                log_file.write("[{}] {}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Exits!"))
                os._exit(0)
                return "Exits!"
            else:
                return "Unkown command!"
        else:
            return "Hello CmdDaemon!"

    return app


def run_web_service(host, port, script):
    app = create_app(script)
    app.run(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--script", type=str, required=True)
    args = parser.parse_args()
    host = args.host
    port = args.port
    script = args.script

    # with context:
    run_web_service(host, port, script)
