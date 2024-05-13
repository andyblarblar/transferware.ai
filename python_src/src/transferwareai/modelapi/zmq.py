import zmq
from zmq.asyncio import Context

zmq_context = None
zmq_pub = None
zmq_sub = None


def initialize_zmq():
    global zmq_context, zmq_pub, zmq_sub
    zmq_context = zmq.asyncio.Context()
    zmq_pub = zmq_context.socket(zmq.PUB)
    zmq_sub = zmq_context.socket(zmq.SUB)

    zmq_sub.connect("tcp://0.0.0.0:5555")
    zmq_pub.bind("tcp://*:5555")
    # TODO I dont think zmq works the way I think lol, could just use mqtt with a broker
