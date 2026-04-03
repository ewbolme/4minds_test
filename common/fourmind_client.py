import asyncio
import hashlib
import json
import os

import requests
import websockets

BASE_URL = "https://api.4minds.ai/api/v1/user"
WS_URL = "wss://api.4minds.ai/api/v1/user/ws"


def _http_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['FOURMIND_API_KEY']}",
        "Content-Type": "application/json",
    }


def _create_conversation(model_id: int, timeout: int = 30) -> str:
    """Create a fresh conversation and return its ID."""
    r = requests.post(
        f"{BASE_URL}/conversations",
        headers=_http_headers(),
        json={"title": "eval", "model_id": model_id},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["id"]


async def _query_ws(conv_id: str, question: str, model_id: int, timeout: int) -> dict:
    """
    Connect via WebSocket, send a conversation_prompt, collect stream_chunks
    until the complete message arrives, then return the result dict.
    """
    api_key = os.environ["FOURMIND_API_KEY"]
    chunks: dict[int, str] = {}
    complete_data: dict = {}

    async with websockets.connect(
        WS_URL,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=30,
        close_timeout=10,
    ) as ws:
        payload = {
            "type": "conversation_prompt",
            "prompt": question,
            "user_message": question,
            "conversation_id": conv_id,
            "model_id": model_id,
            "enable_web_search": False,
        }
        # Step 1: authenticate
        fingerprint = hashlib.md5(api_key.encode()).hexdigest()
        await ws.send(json.dumps({
            "type": "authenticate",
            "token": api_key,
            "fingerprint": fingerprint,
        }))

        # Step 2: wait for authenticated + connection_established before querying
        async with asyncio.timeout(30):
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "auth_error":
                    raise ValueError(f"WebSocket auth failed: {data.get('message')}")
                if msg_type == "connection_established":
                    break  # ready to send

        # Step 3: send the query
        await ws.send(json.dumps(payload))

        async with asyncio.timeout(timeout):
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong", "timestamp": data.get("timestamp")}))

                elif msg_type == "heartbeat":
                    await ws.send(json.dumps({"type": "heartbeat_ack", "timestamp": data.get("timestamp")}))

                elif msg_type == "stream_chunk":
                    seq = data.get("seq", 0)
                    chunks[seq] = data.get("data", "")

                elif msg_type == "complete":
                    complete_data = data
                    break

    answer = "".join(v for _, v in sorted(chunks.items()))
    metadata = complete_data.get("metadata", {})

    return {
        "answer": answer,
        "total_tokens": metadata.get("total_tokens", complete_data.get("total_tokens", 0)),
        "processing_time_ms": metadata.get("processing_time_ms", complete_data.get("processing_time_ms", 0)),
        "context_chunks": metadata.get("context_chunks", complete_data.get("context_chunks", 0)),
        "conversation_id": conv_id,
    }


def query(question: str, model_id: int, timeout: int = 120) -> dict:
    """
    Send a single question to the 4minds model via WebSocket and return:
      {
        "answer":             str,
        "total_tokens":       int,
        "processing_time_ms": int,
        "context_chunks":     int,
        "conversation_id":    str,
      }
    """
    conv_id = _create_conversation(model_id)
    return asyncio.run(_query_ws(conv_id, question, model_id, timeout))
