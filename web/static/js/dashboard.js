// WebSocket kết nối nhận cảnh báo thời gian thực
const alertFeed = document.getElementById("alertFeed");
const alertCountEl = document.getElementById("alertCount");
let totalAlerts = 0;
const alertCountsByType = { INTRUSION: 0, LOITERING: 0, RUNNING: 0, FALLING: 0 };

// WebSocket có tự động kết nối lại.
let ws = null;
let reconnectTimer = null;
let reconnectDelay = 1000;

function connectAlertSocket() {
    if (
        ws &&
        (ws.readyState === WebSocket.OPEN ||
         ws.readyState === WebSocket.CONNECTING)
    ) {
        return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(
        `${protocol}//${window.location.host}/ws/alerts`
    );
    ws = socket;

    socket.onopen = () => {
        reconnectDelay = 1000;
        console.log("[WS] Đã kết nối kênh cảnh báo.");
    };

    socket.onmessage = (event) => {
        console.log("[WS] Nhận cảnh báo:", event.data);

        try {
            const data = JSON.parse(event.data);
            addAlertCard(data);
            incrementAlertCounters(data.event_type);
            playBeepSound();
        } catch (error) {
            console.error(
                "[WS] Không hiển thị được cảnh báo:",
                error,
                event.data
            );
        }
    };

    socket.onerror = (error) => {
        console.error("[WS] Lỗi kết nối:", error);
        socket.close();
    };

    socket.onclose = (event) => {
        console.warn(
            `[WS] Mất kết nối (${event.code}), đang thử lại...`
        );

        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(
            connectAlertSocket,
            reconnectDelay
        );

        reconnectDelay = Math.min(reconnectDelay * 2, 10000);
    };
}

connectAlertSocket();

function renderAlertCounters() {
    alertCountEl.innerText = totalAlerts;
    for (const [type, count] of Object.entries(alertCountsByType)) {
        const el = document.getElementById(`count${type}`);
        if (el) el.innerText = count;
    }
}

function incrementAlertCounters(eventType) {
    totalAlerts++;
    alertCountsByType[eventType] = (alertCountsByType[eventType] || 0) + 1;
    renderAlertCounters();
}

function addAlertCard(data) {
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
}

// Nạp lại lịch sử cảnh báo và bộ đếm từ SQLite khi tải (F5) trang
async function loadAlertHistory() {
    try {
        const [eventsRes, statsRes] = await Promise.all([
            fetch("/api/events"),
            fetch("/api/events/stats")
        ]);
        const events = await eventsRes.json();
        const stats = await statsRes.json();

        // API trả về mới nhất trước, còn addAlertCard chèn lên đầu nên duyệt ngược
        for (const ev of events.slice().reverse()) {
            addAlertCard(ev);
        }

        totalAlerts = stats.total || 0;
        Object.assign(alertCountsByType, stats.by_type || {});
        renderAlertCounters();
    } catch (e) {
        console.error("Lỗi khi tải lịch sử cảnh báo:", e);
    }
}

// Không dùng window.onload: thẻ <img> MJPEG stream không bao giờ tải xong nên sự kiện load không bao giờ xảy ra
loadAlertHistory();

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

// Kích thước gốc của khung hình video do backend trả về (tọa độ OpenCV)
function getFrameSize() {
    return {
        width: videoStream.naturalWidth || 1280,
        height: videoStream.naturalHeight || 720
    };
}

function toggleDrawROI() {
    isDrawing = !isDrawing;
    const btn = document.getElementById("btnDrawROI");
    if (isDrawing) {
        canvas.style.display = "block";
        polygonPoints = [];
        syncCanvasToVideo();
        btn.innerText = "❌ Hủy vẽ";
        btn.classList.replace("btn-primary", "btn-danger");
    } else {
        canvas.style.display = "none";
        btn.innerText = "🖌️ Vẽ vùng ROI mới";
        btn.classList.replace("btn-danger", "btn-primary");
    }
}

// Phủ canvas khít lên vùng ảnh video thực (ảnh được căn giữa trong khung đen nên không trùng khung bao)
function syncCanvasToVideo() {
    const wrapperRect = videoStream.parentElement.getBoundingClientRect();
    const videoRect = videoStream.getBoundingClientRect();
    canvas.style.left = `${videoRect.left - wrapperRect.left}px`;
    canvas.style.top = `${videoRect.top - wrapperRect.top}px`;
    canvas.style.width = `${videoRect.width}px`;
    canvas.style.height = `${videoRect.height}px`;
    canvas.width = Math.round(videoRect.width);
    canvas.height = Math.round(videoRect.height);
    redrawCanvas();
}

window.addEventListener("resize", () => {
    if (isDrawing) syncCanvasToVideo();
});

canvas.addEventListener("click", (e) => {
    if (!isDrawing) return;
    // Tính theo khung ảnh video để không phụ thuộc vị trí canvas
    const rect = videoStream.getBoundingClientRect();
    const frameSize = getFrameSize();
    const x = (e.clientX - rect.left) * frameSize.width / rect.width;
    const y = (e.clientY - rect.top) * frameSize.height / rect.height;
    if (x < 0 || y < 0 || x > frameSize.width || y > frameSize.height) return;

    polygonPoints.push([Math.round(x), Math.round(y)]);
    syncCanvasToVideo();
});

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (polygonPoints.length === 0) return;

    const frameSize = getFrameSize();
    const scaleX = canvas.width / frameSize.width;
    const scaleY = canvas.height / frameSize.height;

    const isLoiteringZone = document.getElementById("zoneTypeSelect").value === "LOITERING_ZONE";
    ctx.strokeStyle = isLoiteringZone ? "#eab308" : "#ef4444";
    ctx.fillStyle = isLoiteringZone ? "rgba(234, 179, 8, 0.2)" : "rgba(239, 68, 68, 0.2)";
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

    const zoneType = document.getElementById("zoneTypeSelect").value;
    const defaultName = zoneType === "LOITERING_ZONE" ? "Vùng lảng vảng mới" : "Vùng cấm mới";
    const zoneName = prompt("Nhập tên vùng mới:", defaultName);
    if (!zoneName) return;

    const newZone = {
        id: "ZONE_" + Date.now().toString().slice(-4),
        name: zoneName,
        type: zoneType,
        polygon: polygonPoints
    };

    try {
        const response = await fetch("/api/zones", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(newZone)
        });
        if (response.ok) {
            alert("Đã lưu và kích hoạt vùng ROI mới thành công!");
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

// -------------------------------------------------------------
// CHUYỂN ĐỔI NGUỒN VIDEO & UPLOAD VIDEO NGAY TRÊN GIAO DIỆN
// -------------------------------------------------------------
async function onSelectVideoSource(sourceValue) {
    try {
        const response = await fetch("/api/change_source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: sourceValue })
        });
        if (response.ok) {
            console.log("[INFO] Đã chuyển nguồn video sang:", sourceValue);
            // Tải lại thẻ img video_feed để làm mới luồng stream
            refreshVideoStream();
        }
    } catch (e) {
        console.error("Lỗi khi chuyển nguồn video:", e);
    }
}

async function uploadCustomVideo(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const btn = input.parentElement;
    const oldText = btn.innerHTML;
    btn.innerText = "⏳ Đang tải...";

    try {
        const res = await fetch("/api/upload_video", {
            method: "POST",
            body: formData
        });
        if (res.ok) {
            const data = await res.json();
            alert(`Tải video ${file.name} thành công! Hệ thống đang phát video của bạn.`);
            
            // Thêm vào dropdown menu và chọn
            const select = document.getElementById("videoSelect");
            const opt = document.createElement("option");
            opt.value = data.video_path;
            opt.innerText = `🎬 ${file.name}`;
            select.appendChild(opt);
            select.value = data.video_path;
            
            refreshVideoStream();
        }
    } catch (e) {
        alert("Lỗi khi tải video: " + e);
    } finally {
        btn.innerHTML = oldText;
    }
}

function refreshVideoStream() {
    const img = document.getElementById("videoStream");
    const currentSrc = img.src.split("?")[0];
    img.src = `${currentSrc}?t=${Date.now()}`;
}

