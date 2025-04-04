import Toastify from "toastify-js";
import "toastify-js/src/toastify.css"

export default function showToast(type, message, callback_func) {
  let config = {
    text: message,
    duration: 2000,
    gravity: "top",
    position: "right",
    close: true,
    stopOnFocus: true,
    style: {
      color: "white",
      borderRadius: "8px",
      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
      fontWeight: "bold",
      padding: "12px 16px",
      fontSize: "14px",
    },
    callback: function (){
      if (!callback_func) return;
      if (callback_func) {
        callback_func()
      }
    }
  };

  switch (type) {
    case "info":
      config.text = "ℹ️ " + message;
      config.style.background = "#3498db"; // 蓝色
      break;
    case "warning":
      config.text = "⚠️ " + message;
      config.style.background = "#f39c12"; // 橙黄色
      break;
    case "error":
      config.text = "❌ " + message;
      config.style.background = "#e74c3c"; // 红色
      break;
    case "success":
      config.text = "✅ " + message;
      config.style.background = "#2ecc71"; // 绿色
      break;
    default:
      config.text = message;
      config.style.background = "black"; // 默认黑色
  }

  Toastify(config).showToast();
}
