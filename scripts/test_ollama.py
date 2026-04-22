"""Quick smoke test of the Ollama model - text + image."""
import sys
import time
import base64
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


MODEL = "qwen3.5:4b"


def test_text() -> None:
    print(f"\n[TEST 1] Text with {MODEL}")
    llm = ChatOllama(model=MODEL, temperature=0.7)
    messages = [
        SystemMessage(content="You are a friendly 24-year-old. Reply short, informal, in English."),
        HumanMessage(content="Hi, how's it going?"),
    ]
    start = time.time()
    resp = llm.invoke(messages)
    elapsed = time.time() - start
    print(f"  Reply: {resp.content}")
    print(f"  Time: {elapsed:.2f}s")


def test_vision(image_path: str | None = None) -> None:
    print(f"\n[TEST 2] Image with {MODEL}")
    if image_path is None or not Path(image_path).exists():
        print("  (skipping: no test image - pass a path as argv[1] to test)")
        return

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    llm = ChatOllama(model=MODEL, temperature=0.7)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "What do you see in this photo? Reply short and natural."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"},
        ]
    )
    start = time.time()
    resp = llm.invoke([msg])
    elapsed = time.time() - start
    print(f"  Reply: {resp.content}")
    print(f"  Time: {elapsed:.2f}s")


if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_text()
    test_vision(image_arg)
    print("\nTest complete.")
