import sys
import json
from client import SoftActorCriticEntropy

sac = SoftActorCriticEntropy(0.2)

def handle_call(name, arguments):
    if name == "compute":
        q = arguments["q_values"]
        lp = arguments["log_probs"]
        a = arguments.get("alpha", 0.2)
        inst = SoftActorCriticEntropy(a)
        return {"objective": inst.compute_objective(q, lp)}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
