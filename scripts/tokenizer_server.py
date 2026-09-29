"""CPU tokenizer endpoint for an inference backend without /tokenize.

Run inside the server model's transformers environment. Bind loopback only.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from transformers import AutoTokenizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model-path",required=True);parser.add_argument("--served-model-name",required=True)
    parser.add_argument("--port",type=int,default=18159);args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=False)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path!="/tokenize":raise ValueError("unknown endpoint")
                length=int(self.headers.get("Content-Length",0))
                if length>4*1024*1024:raise ValueError("request too large")
                data=json.loads(self.rfile.read(length))
                if data.get("model")!=args.served_model_name:raise ValueError("model name mismatch")
                tokens=tokenizer.encode(data["prompt"],add_special_tokens=False)
                payload=json.dumps({"count":len(tokens),"tokens":tokens}).encode();self.send_response(200)
            except Exception as error:
                payload=json.dumps({"error":str(error)}).encode();self.send_response(400)
            self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()


if __name__=="__main__":main()
