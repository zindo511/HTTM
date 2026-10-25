// WebSocket kết nối nhận cảnh báo thời gian thực
const alertFeed = document.getElementById("alertFeed");
const alertCountEl = document.getElementById("alertCount");
let totalAlerts = 0;

// Khởi tạo WebSocket
const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/alerts`);

ws.onopen = () => {
    console.log("[WS] Đã kết nối thành công tới WebSocket Alert Channel.");
};

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        addAlertCard(data);
    } catch (e) {
        console.error("Lỗi parse dữ liệu WebSocket:", e);
    }
};

function addAlertCard(data) {
    totalAlerts++;
    alertCountEl.innerText = totalAlerts;

    const card = document.createElement("div");
    card.className = `alert-card ${data.event_type}`;
    
    let icon = "⚠️";
    if (data.event_type === "INTRUSION") icon = "🚫";
    if (data.event_type === "LOITERING") icon = "⏳";
    if (data.event_type === "FALLING") icon = "🚨";
    if (data.event_type === "RUNNING") icon = "🏃";

    card.innerHTML = `
        <img class="alert-thumb" src="${data.snapshot_url}" alt="Snapshot" onerror="this.src='/static/placeholder.png'">
        <div class="alert-info">
            <div class="alert-title">
                <span>${icon} ${data.event_type} (Track #${data.track_id})</span>
                <span style="font-size: 0.75rem; color: #38bdf8;">${data.confidence}%</span>
            </div>
            <div class="alert-time">${data.timestamp} | Vùng: ${data.zone_id}</div>
            <div class="alert-desc">${data.extra_info || ""}</div>
        </div>
    `;

    alertFeed.insertBefore(card, alertFeed.firstChild);
    
    // Tự động phát âm thanh cảnh báo ngắn
    playBeepSound();
}

function playBeepSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.frequency.value = 880;
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        // Trình duyệt có thể chặn autoplay
    }
}

// -------------------------------------------------------------
// LOGIC VẼ ĐA GIÁC VÙNG CẤM ROI BẰNG CHUỘT
// -------------------------------------------------------------
const canvas = document.getElementById("roiCanvas");
const ctx = canvas.getContext("2d");
const videoStream = document.getElementById("videoStream");
let isDrawing = false;
let polygonPoints = [];

function toggleDrawROI() {
    isDrawing = !isDrawing;
    const btn = document.getElementById("btnDrawROI");
    if (isDrawing) {
        canvas.style.display = "block";
        canvas.width = videoStream.clientWidth;
        canvas.height = videoStream.clientHeight;
        polygonPoints = [];
        redrawCanvas();
        btn.innerText = "❌ Hủy vẽ";
        btn.classList.replace("btn-primary", "btn-danger");
    } else {
        canvas.style.display = "none";
        btn.innerText = "🖌️ Vẽ vùng ROI mới";
        btn.classList.replace("btn-danger", "btn-primary");
    }
}

canvas.addEventListener("click", (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    
    // Quy đổi tỷ lệ nếu video hiển thị co giãn
    const scaleX = 1280 / canvas.width;
    const scaleY = 720 / canvas.height;
    
    polygonPoints.push([Math.round(x * scaleX), Math.round(y * scaleY)]);
    redrawCanvas();
});

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (polygonPoints.length === 0) return;

    const scaleX = canvas.width / 1280;
    const scaleY = canvas.height / 720;

    ctx.strokeStyle = "#ef4444";
    ctx.fillStyle = "rgba(239, 68, 68, 0.2)";
    ctx.lineWidth = 2;

    ctx.beginPath();
    const [startX, startY] = polygonPoints[0];
    ctx.moveTo(startX * scaleX, startY * scaleY);

    for (let i = 1; i < polygonPoints.length; i++) {
        const [px, py] = polygonPoints[i];
        ctx.lineTo(px * scaleX, py * scaleY);
    }

    if (polygonPoints.length >= 3) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();

    // Vẽ chấm tròn tại các đỉnh
    ctx.fillStyle = "#ffffff";
    for (const [px, py] of polygonPoints) {
        ctx.beginPath();
        ctx.arc(px * scaleX, py * scaleY, 4, 0, Math.PI * 2);
        ctx.fill();
    }
}

async function saveROI() {
    if (polygonPoints.length < 3) {
        alert("Vui lòng nhấp chuột chọn ít nhất 3 điểm để tạo đa giác!");
        return;
    }

    const zoneName = prompt("Nhập tên vùng cấm mới:", "Vùng cấm mới");
    if (!zoneName) return;

    const newZone = {
        id: "ZONE_" + Date.now().toString().slice(-4),
        name: zoneName,
        type: "RESTRICTED_INTRUSION",
        polygon: polygonPoints
    };

    try {
        const response = await fetch("/api/zones", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(newZone)
        });
        if (response.ok) {
            alert("Đã lưu và kích hoạt vùng cấm ROI mới thành công!");
            toggleDrawROI();
        }
    } catch (e) {
        alert("Lỗi khi lưu vùng ROI: " + e);
    }
}

async function clearROI() {
    if (!confirm("Bạn có chắc chắn muốn xóa tất cả các vùng cấm hiện tại?")) return;
    try {
        await fetch("/api/zones/clear", { method: "POST" });
        alert("Đã xóa sạch các vùng ROI.");
    } catch (e) {
        alert("Lỗi khi xóa vùng: " + e);
    }
}
