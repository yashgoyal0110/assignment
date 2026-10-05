"""Interactive demo: type a customer message, see the full agent decision.

  python scripts/demo.py
  python scripts/demo.py "my parcel never arrived"
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import Agent


def show(rec):
    print(f"\n  intent     : {rec['intent']}  (conf {rec['confidence']:.2f})")
    print(f"  decision   : {rec['decision'].upper()}")
    print(f"  reason     : {rec['decision_reason']}")
    print(f"  top match  : [{rec['retrieved'][0]['similarity']:.2f}] "
          f"{rec['retrieved'][0]['customer_text'][:60]}")
    print(f"  draft reply: {rec['reply']}\n")


def main():
    agent = Agent()
    if len(sys.argv) > 1:
        show(agent.handle(" ".join(sys.argv[1:])))
        return
    print("Type a customer message (blank to quit).")
    while True:
        try:
            msg = input("\n> ").strip()
        except EOFError:
            break
        if not msg:
            break
        show(agent.handle(msg))


if __name__ == "__main__":
    main()
