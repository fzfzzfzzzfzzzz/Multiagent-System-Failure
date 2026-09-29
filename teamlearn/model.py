from __future__ import annotations

import json
import os
import re
import time
import httpx
from .common import canonical
from .progress import progress


class ModelError(RuntimeError): pass


def parse_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict): raise ValueError("expected one JSON object")
    return value


class Model:
    def __init__(self, config, ledger=None):
        self.config = config
        self.name = config["model"]
        self.ledger = ledger if ledger is not None else []
        self.http = httpx.Client(base_url=config["base_url"].rstrip("/"), timeout=config.get("timeout", 120), trust_env=False,
                                 headers={"Authorization": "Bearer " + os.getenv(config.get("api_key_env", "TEAMLEARN_API_KEY"), "EMPTY")})
        self.token_cache = {}

    def tokens(self, text):
        if not text: return 0
        if text not in self.token_cache:
            # Fail closed; whitespace or byte estimates cannot certify an equal-token experiment.
            url = self.config.get("tokenizer_url") or self.config["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"
            response = self.http.post(url, json={"model": self.name, "prompt": text, "add_special_tokens": False})
            response.raise_for_status()
            data = response.json()
            self.token_cache[text] = data.get("count", len(data.get("tokens", data.get("input_ids", []))))
            if not self.token_cache[text]: raise ModelError("tokenizer endpoint returned no count")
        return self.token_cache[text]

    def complete(self, messages, seed, phase="execution", max_tokens=None, schema=None):
        body = {"model": self.name, "messages": messages, "temperature": self.config.get("temperature", 0.2),
                "max_tokens": max_tokens or self.config.get("max_tokens", 512), "seed": seed,
                **self.config.get("extra_body", {})}
        if schema is not None:
            body["response_format"]={"type":"json_schema","json_schema":{"name":"teamlearn_response","schema":schema}}
        start = time.monotonic()
        # Retry transport errors only; malformed generations count as actual attempts.
        for attempt in range(3):
            try:
                response = self.http.post("/chat/completions", json=body)
                if response.status_code in (429, 502, 503, 504):
                    self.ledger.append({"model": self.name, "phase": phase, "attempt": attempt, "transport_error": response.status_code,
                                        "usage_unknown": True, "seconds": time.monotonic()-start})
                    if attempt == 2: response.raise_for_status()
                    time.sleep(1 + attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                if data.get("model")!=self.name:raise ModelError(f"server returned a different model: {data.get('model')}")
                usage = data.get("usage")
                if not usage or "prompt_tokens" not in usage: raise ModelError("API must report real token usage")
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                self.ledger.append({"model": self.name, "phase": phase, "seed": seed, "attempt": attempt,
                                    "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                                    "seconds": time.monotonic()-start, "finish_reason": choice.get("finish_reason"),
                                    "system_fingerprint": data.get("system_fingerprint"), "response_id": data.get("id")})
                progress('model_call_completed',model=self.name,phase=phase,input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'])
                return text
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                self.ledger.append({"model": self.name, "phase": phase, "attempt": attempt, "transport_error": type(error).__name__, "usage_unknown": True})
                if attempt == 2: raise ModelError(str(error)) from error
                time.sleep(1 + attempt)
        raise ModelError("request failed")

    def close(self): self.http.close()


class MockModel:
    """Deliberately simple action generator to exercise I/O, never a baseline model."""
    name = "mock-NOT-RESEARCH"
    def __init__(self, config=None, ledger=None):
        self.config = config or {}
        self.ledger = ledger if ledger is not None else []

    def tokens(self, text): return len(re.findall(r"\w+|[^\w\s]", text))

    def complete(self, messages, seed, phase="execution", max_tokens=None, schema=None):
        if phase == "report":
            payload = json.loads(messages[-1]["content"])
            refs = [e["event_id"] for e in payload["events"]]
            obj = {"components": {c: [] for c in "S OCDKPVU".replace(" ", "")},
                   "predicted_mechanism": "unknown", "culprits": [], "related_roles": ["provider", "analyst"],
                   "source_agents": payload.get("agents", {})}
            obj["components"]["O"] = [{"text": "Source execution requires review.", "refs": refs[:1], "status": "observed", "roles": list(payload.get("agents", {}))}]
            obj["components"]["P"] = [{"text": "Check the current contract before consuming an artifact.", "refs": refs[:1], "status": "proposed_prevention", "roles": ["analyst", "executor"]}]
        else:
            payload = json.loads(messages[-1]["content"])
            role = payload["role"]
            history = payload["own_events"]
            ops = [e["action"].get("tool") for e in history]
            def latest(op):
                return next((e["observation"] for e in reversed(history) if e["action"].get("tool") == op and not e["observation"].get("error")), {})
            op, args = "finish", {}
            if "contract" not in ops: op = "contract"
            elif role == "coordinator":
                if "assign" not in ops: op,args = "assign",{"check":"completion","role":"executor"}
                elif "signoff" not in ops: op = "signoff"
                elif "message" not in ops: op,args = "message",{"to":"provider","text":"Use the smaller allocation and confirm asynchronous publication."}
            elif role == "provider":
                if "read_records" not in ops: op = "read_records"
                elif "publish" not in ops:
                    data=latest("read_records"); op,args="publish",{k:data[k] for k in ("values","version","unit")}
                elif "allocate" not in ops: op,args="allocate",{"amount":2}
                elif latest("contract").get("confirmation") == "receipt_required" and "confirm" not in ops: op="confirm"
            elif role == "analyst":
                if "read_artifact" not in ops: op="read_artifact"
                elif "calculate" not in ops:
                    data=latest("read_artifact").get("artifact")
                    if data:
                        from .datasets import compute
                        ct=latest("contract");op,args="calculate",{"value":compute(ct["rule"],data["values"],ct["weights"]),"version":data["version"]}
                elif "allocate" not in ops: op,args="allocate",{"amount":3}
            elif role == "executor":
                if "read_artifact" not in ops: op="read_artifact"
                elif "verify" not in ops: op,args="verify",{"source":"independent"}
                elif "commit" not in ops: op="commit"
            obj = {"tool": op, "args": args}
        result = canonical(obj)
        self.ledger.append({"model":self.name,"phase":phase,"input_tokens":self.tokens(canonical(messages)),"output_tokens":self.tokens(result),"seconds":0,"mock":True})
        return result

    def close(self): pass


def make_model(config, ledger=None):
    return MockModel(config, ledger) if config.get("backend") == "mock" else Model(config, ledger)
