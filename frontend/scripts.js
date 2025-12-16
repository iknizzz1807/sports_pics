const fileInput = document.getElementById("file-input");
const imgPreview = document.getElementById("image-preview");
const placeholder = document.getElementById("placeholder-text");
const resultArea = document.getElementById("result-area");
const predictBtn = document.getElementById("predict-btn");
const spinner = document.getElementById("loading-spinner");

let currentFile = null;

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
      resultArea.style.display = "none";
    };
    reader.readAsDataURL(file);
  }
});

async function handlePredict() {
  if (!currentFile) {
    alert("Vui lòng chọn ảnh trước!");
    return;
  }

  // UI: Bật loading
  predictBtn.disabled = true;
  predictBtn.innerText = "Đang xử lý...";
  spinner.style.display = "block";
  resultArea.style.display = "none";

  try {
    const formData = new FormData();
    formData.append("image", currentFile);
    // Không cần append method nữa

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    displayResult(data);
  } catch (error) {
    console.error("Lỗi:", error);
    alert("Có lỗi xảy ra: " + error.message);
  } finally {
    // UI: Reset
    predictBtn.disabled = false;
    predictBtn.innerText = "Phân loại ngay";
    spinner.style.display = "none";
  }
}

function displayResult(data) {
  document.getElementById("res-label").innerText = data.label;

  // Backend trả về method, hoặc ta hardcode hiển thị

  const confidenceVal = data.confidence;
  const percent = (confidenceVal * 100).toFixed(2) + "%";

  document.getElementById("res-conf").innerText = percent;

  const bar = document.getElementById("conf-bar");
  bar.style.width = percent;

  if (confidenceVal > 0.8) {
    bar.style.backgroundColor = "#10b981";
  } else if (confidenceVal > 0.6) {
    bar.style.backgroundColor = "#f59e0b";
  } else {
    bar.style.backgroundColor = "#ef4444";
  }

  resultArea.style.display = "block";
}
