import entangle
import json
import datetime
from ailab.server import Server


def run_server(config_path):
    with open("exceptions.log", "w") as log:
        log.write("%s: Server Started\n" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    config = {}
    with open(config_path, "r") as f:
        config = json.loads(f.read())

    server = Server(config, config_path)

    # Listen for entanglements (listenes in blocking mode)
    entangle.listen(host=config["host"], port=config["port"], users=config["users"], callback=server.on_entangle)

    server.running = False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_server(sys.argv[1])
    else:
        run_server("config/default.json")
