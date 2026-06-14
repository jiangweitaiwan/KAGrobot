# -*- coding: utf-8 -*-
"""entry.py — 服務啟動點。"""

import uvicorn
from app import create_app

app = create_app()

if __name__ == "__main__":
    from settings import get_settings
    s = get_settings()
    uvicorn.run("entry:app", host=s.host, port=s.port, reload=False)
