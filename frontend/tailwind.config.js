export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Fredoka", "ui-rounded", "Nunito", "system-ui", "sans-serif"],
      },
      colors: {
        foam: "#fbfeff",
        deep: "#296f9d",
        lagoon: "#8bdff2",
        pearl: "#f7fdff",
        coral: {
          DEFAULT: "#ff7f91",
          light: "#ffb3bf",
        },
      },
      boxShadow: {
        pearl: "0 18px 60px rgba(184, 239, 255, 0.42)",
        coral: "0 18px 54px rgba(255, 127, 145, 0.34)",
      },
    },
  },
  plugins: [],
};
