import flask
from flask import Flask, request, Response
from flask import request
import time
from flask_cors import CORS,cross_origin
import gumball_lite_environ as env

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

gumball_envs = {}
envs_last_used_timers = {}

BOT_COUNT = 15
for i in range(BOT_COUNT):
    bot_id = str(i)
    gumball_envs[bot_id] = env.make_env()
    envs_last_used_timers[bot_id] = time.time()

import numpy as np
import os
import json

FLASK_PORT = int(os.environ.get('FLASK_PORT', 12231))
DEBUG = bool(int(os.environ.get('DEBUG', 1)))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


@app.route('/api/reset',methods=['POST', 'OPTIONS'])
@cross_origin()
def reset_env():
    print('enter reset_env')
    payload = {}

     # let's say how long to stay on
    params = request.get_json()

    bot_id = params['bot_id']
    # reset last used timer
    envs_last_used_timers[bot_id] = time.time()

    env = gumball_envs[bot_id]

    obs, reward, done, info = env.reset()
    resp = {
        'obs': obs,
        'reward': reward,
        'done': done,
        'info': info
    }

    print(obs)
    print('exit step', resp)
    return json.dumps(resp, cls=NpEncoder)

@app.route('/api/step',methods=['POST', 'OPTIONS'])
@cross_origin()
def step():
    print('enter step')
    payload = {}

     # let's say how long to stay on
    params = request.get_json()

    bot_id = params['bot_id']
    env = gumball_envs[bot_id]

    action_as_int = params['action']

    try:
        obs, reward, done, info = env.step(action_as_int)
        resp = {
            'obs': obs,
            'reward': reward,
            'done': done,
            'info': info
        }
    except Exception as e:
        print(e)
        raise e

#    return jsonify(resp)
#    print(resp)
#    print('exit step', resp)
    return json.dumps(resp, cls=NpEncoder)

@app.route('/api/get_next_bot_id',methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def get_next_bot_id():
    print('enter get_next_bot_id')
    payload = {}

    bot_id = 0
    min_time = envs_last_used_timers["0"]

    for i in range(BOT_COUNT):
        new_time = envs_last_used_timers[str(i)]
        if new_time < min_time:
            min_time = new_time
            bot_id = i

    resp = {
        'bot_id':str(bot_id)
    }

    print('exit get_next_bot_id', resp)
    return json.dumps(resp, cls=NpEncoder)


@app.route('/api/get_sample_action',methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def get_sample_action():
    payload = {}

    params = request.get_json()
    bot_id = params['bot_id']
    env = gumball_envs[bot_id]

    action_as_int = env.sample_actions()
    resp = {
        'action':action_as_int,
    }

    return json.dumps(resp, cls=NpEncoder)


if __name__ == '__main__':
    app.run(debug=DEBUG,host='0.0.0.0', port=FLASK_PORT, threaded=True)


def jsonify(obj, status=200, headers=None):
    """ Custom JSONificaton to support obj.to_dict protocol. """
    data = NpEncoder().encode(obj)
    if 'callback' in request.args:
        cb = request.args.get('callback')
        data = '%s && %s(%s)' % (cb, cb, data)
    return Response(data, headers=headers, status=status,
                    mimetype='application/json')
