# -*- coding: utf-8 -*-
"""backend/core — 依職責分層的共用領域邏輯。

子套件
  infra     Firebase / Firestore、LRU cache、widget JWT
  llm       Gemini / Vertex 工廠與重試
  account   會員等級、用量統計、額度檢查
  sessions  對話 session 與 RC briefing 持久化
  skills    Skill 載入、e2b sandbox、compute 端點
  agents    聊天狀態、Socrates / Main / Walk bot 與 walk SSE
  brain     語意上下文、RC briefing 生成
  admin     管理者目錄、Firestore 彙總、稽核紀錄
"""
