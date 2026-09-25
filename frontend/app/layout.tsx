import "./globals.css";
import type { Metadata } from "next";
import { AuthProvider } from "./components/AuthProvider";

export const metadata: Metadata = {
  title: "ZEN",
  description: "ZEN — 從感知到行動的因果引擎",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
