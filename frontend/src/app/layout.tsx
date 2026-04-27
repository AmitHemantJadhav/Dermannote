import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DermAnnotate",
  description:
    "ML-Assisted Medical Image Annotation for Autoimmune Skin Conditions",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
