"""CPU-only fixtures for testing remote process cleanup; never experiments."""
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import time

parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['server','success','failure','idle']);parser.add_argument('--port',type=int,default=18991)
args=parser.parse_args()
if args.mode=='server':
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload=json.dumps({'data':[{'id':'cleanup-test'}]}).encode();self.send_response(200);self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    HTTPServer.allow_reuse_address=True
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
elif args.mode=='success':time.sleep(.5)
elif args.mode=='failure':raise SystemExit(7)
else:time.sleep(120)
