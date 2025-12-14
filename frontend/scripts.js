// File: scripts.js

// --- 1. XỬ LÝ GIAO DIỆN (PREVIEW ẢNH) ---
const fileInput = document.getElementById("file-input");
const imgPreview = document.getElementById("image-preview");
const placeholder = document.getElementById("placeholder-text");
const resultArea = document.getElementById("result-area");
const predictBtn = document.getElementById("predict-btn");
const spinner = document.getElementById("loading-spinner");

let currentFile = null;

// API Endpoint (Đảm bảo backend chạy ở port 8000)
const API_URL = "http://localhost:8000/api/predict";

fileInput.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = function (e) {
      imgPreview.src = e.target.result;
      imgPreview.style.display = "block";
      placeholder.style.display = "none";
      // Reset kết quả cũ
      resultArea.style.display = "none";
    };
    reader.readAsDataURL(file);
  }
});

// --- 2. LOGIC GỌI API ---

async function handlePredict() {
  if (!currentFile) {
    alert("Vui lòng chọn ảnh trước!");
    return;
  }

  const method = document.getElementById("method-select").value;

  // UI: Bật loading, tắt nút
  predictBtn.disabled = true;
  predictBtn.innerText = "Đang xử lý...";
  spinner.style.display = "block";
  resultArea.style.display = "none";

  try {
    // Tạo form data gửi lên backend
    const formData = new FormData();
    formData.append("image", currentFile);
    formData.append("method", method);

    // Gọi API thật
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Hiển thị kết quả
    displayResult(data);
  } catch (error) {
    console.error("Lỗi:", error);
    alert("Có lỗi xảy ra khi xử lý: " + error.message);
  } finally {
    // UI: Reset trạng thái
    predictBtn.disabled = false;
    predictBtn.innerText = "Phân loại ngay";
    spinner.style.display = "none";
  }
}

// --- 3. HÀM HIỂN THỊ KẾT QUẢ ---
function displayResult(data) {
  document.getElementById("res-label").innerText = data.label;
  document.getElementById("res-method").innerText = data.method;

  // Format phần trăm
  const confidenceVal = data.confidence;
  const percent = (confidenceVal * 100).toFixed(2) + "%";

  document.getElementById("res-conf").innerText = percent;

  const bar = document.getElementById("conf-bar");
  bar.style.width = percent;

  // Đổi màu thanh confidence dựa trên độ tin cậy
  if (confidenceVal > 0.8) {
    bar.style.backgroundColor = "#10b981"; // Green
  } else if (confidenceVal > 0.6) {
    bar.style.backgroundColor = "#f59e0b"; // Orange
  } else {
    bar.style.backgroundColor = "#ef4444"; // Red
  }

  resultArea.style.display = "block";
}
