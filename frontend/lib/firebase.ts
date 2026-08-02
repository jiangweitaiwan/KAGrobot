"use client";

/**
 * lib/firebase.ts — Firebase App + Auth 單例。
 *
 * 只初始化 Auth（本專案目前不需要 Firestore/Storage 的前端 SDK，
 * 使用者資料一律經由後端 API 讀寫，見 lib/apiFetch.ts）。
 *
 * getApps().length 檢查是為了在 Next.js dev 模式的 Fast Refresh 下
 * 避免重複 initializeApp() 拋出 "already exists" 錯誤。
 */

import { initializeApp, getApps, getApp, type FirebaseOptions } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig: FirebaseOptions = {
  apiKey:            process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain:        process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId:         process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket:     process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId:             process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

export const firebaseApp = getApps().length ? getApp() : initializeApp(firebaseConfig);
export const auth = getAuth(firebaseApp);
