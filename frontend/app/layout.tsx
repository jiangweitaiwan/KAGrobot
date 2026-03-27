import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "KAGrobot Chat",
  description: "Vercel frontend for KAGrobot backend",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
