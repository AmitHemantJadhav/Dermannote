/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#0f1a1a",
          secondary: "#162424",
          card: "#1c2e2e",
        },
        accent: {
          teal: "#3d8282",
          coral: "#e07a5f",
          gold: "#f2cc8f",
          sage: "#81b29a",
        },
        text: {
          primary: "#e8f0f0",
          secondary: "#8aacac",
          muted: "#5a7d7d",
        },
      },
      fontFamily: {
        heading: ["Georgia", "serif"],
        data: ["Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
