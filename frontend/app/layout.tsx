import "./globals.css";
import type { Metadata } from "next";
import { AuthProvider } from "./components/AuthProvider";

export const metadata: Metadata = {
  title: "KAGrobot Chat",
  description: "Vercel frontend for KAGrobot backend",
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
