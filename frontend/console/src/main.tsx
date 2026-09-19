import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initializeBridge, loadStoredTheme } from "./lib/bridge";
import "./styles.css";

// 渲染前主动应用本地主题记忆，防止首帧闪白后由 App 的 useEffect 二次翻转；
// 无记忆时保持编译期默认主题。宿主令牌经 initializeBridge 的初始化状态
// 路径在用户未接管主题时于渲染后被驱动。
const storedTheme = loadStoredTheme();
if (storedTheme === "dark") document.documentElement.classList.add("dark");
else if (storedTheme === "light") document.documentElement.classList.remove("dark");

initializeBridge();
ReactDOM.createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
