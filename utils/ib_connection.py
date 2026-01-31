from __future__ import annotations
import asyncio
from typing import Optional
from ib_async import IB

# Global IB instance mapping: client_id -> IB instance
_IB_INSTANCES: dict[int, IB] = {}

async def connect_to_ib(host: str = '127.0.0.1', port: int = 7497, client_id: int = 1, symbol: str = "") -> Optional[IB]:
    """
    Connect to IB Gateway or TWS.
    Reuse existing connection if available for the same client_id.
    """
    global _IB_INSTANCES
    
    if client_id in _IB_INSTANCES:
        ib = _IB_INSTANCES[client_id]
        if ib.isConnected():
            return ib
        else:
            # Clean up disconnected instance
            del _IB_INSTANCES[client_id]

    ib = IB()
    try:
        await ib.connectAsync(host, port, client_id)
        _IB_INSTANCES[client_id] = ib
        return ib
    except Exception as e:
        print(f"Error connecting to IB (id={client_id}): {e}")
        return None

async def disconnect_from_ib(ib: IB, symbol: str = "") -> None:
    """
    Disconnect from IB.
    """
    if ib and ib.isConnected():
        ib.disconnect()
        # Remove from global instances if present
        for cid, instance in list(_IB_INSTANCES.items()):
            if instance == ib:
                del _IB_INSTANCES[cid]
                break
