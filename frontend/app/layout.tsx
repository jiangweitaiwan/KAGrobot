import "./globals.css";
import type { Metadata } from "next";
import { AuthProvider } from "./components/AuthProvider";

export const metadata: Metadata = {
  title: "KAGrobot | Knowledge Field",
  description: "Explore a knowledge field, build a personal reasoning foundation, and execute recommended conclusions.",
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
