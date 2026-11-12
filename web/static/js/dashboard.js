// WebSocket kết nối nhận cảnh báo thời gian thực
const alertFeed = document.getElementById("alertFeed");
const alertCountEl = document.getElementById("alertCount");
const alertEmptyState = document.getElementById("alertEmptyState");
const alertEmptyText = document.getElementById("alertEmptyText");
const clearAlertsBtn = document.getElementById("clearAlertsBtn");

let totalAlerts = 0;
const alertCountsByType = { RUNNING: 0, FALLING: 0, FIGHTING: 0 };
let activeFilter = "ALL";
const vnFilterNameMap = {
    ALL: "Tất cả",
    RUNNING: "Chạy nhanh",
    FALLING: "Té ngã",
    FIGHTING: "Ẩu đả",
};

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
            showLiveBanner(data);
            incrementAlertCounters(data.event_type);
            playBeepSound(data.event_type);
            speakAlertCallout(data);
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

function updateEmptyState() {
    if (!alertEmptyState) return;
    const visibleCards = Array.from(alertFeed.children).filter(
        (card) => card.style.display !== "none"
    );
    const isEmpty = visibleCards.length === 0;
    alertEmptyState.style.display = isEmpty ? "flex" : "none";
    alertFeed.style.display = isEmpty ? "none" : "flex";
    if (alertEmptyText) {
        alertEmptyText.innerText =
            activeFilter === "ALL"
                ? "Chưa có cảnh báo nào được ghi nhận"
                : `Không có cảnh báo thuộc mục "${vnFilterNameMap[activeFilter] || activeFilter}"`;
    }
    if (clearAlertsBtn) {
        clearAlertsBtn.disabled = isEmpty && totalAlerts === 0;
        clearAlertsBtn.innerHTML =
            activeFilter === "ALL"
                ? "🗑️ Xoá tất cả"
                : `🗑️ Xoá mục ${vnFilterNameMap[activeFilter]}`;
    }
}

function renderAlertCounters() {
    alertCountEl.innerText = totalAlerts;
    for (const [type, count] of Object.entries(alertCountsByType)) {
        const el = document.getElementById(`count${type}`);
        if (el) el.innerText = count;
        const chipEl = document.getElementById(`chipCount${type}`);
        if (chipEl) chipEl.innerText = count;
    }
    updateEmptyState();
}

function applyStatsFromServer(stats) {
    if (!stats) return;
    totalAlerts = stats.total || 0;
    for (const type of ["RUNNING", "FALLING", "FIGHTING"]) {
        alertCountsByType[type] = (stats.by_type && stats.by_type[type]) || 0;
    }
    renderAlertCounters();
}

function incrementAlertCounters(eventType) {
    totalAlerts++;
    alertCountsByType[eventType] = (alertCountsByType[eventType] || 0) + 1;
    renderAlertCounters();
}

function setAlertFilter(filterType) {
    // Nếu bấm lại bộ lọc đang chọn dưới footer thì quay về ALL
    activeFilter = filterType || "ALL";
    for (const chip of document.querySelectorAll(".filter-chip")) {
        chip.classList.toggle("active", chip.dataset.filter === activeFilter);
    }
    for (const card of alertFeed.children) {
        const match =
            activeFilter === "ALL" || card.dataset.eventType === activeFilter;
        card.style.display = match ? "flex" : "none";
    }
    updateEmptyState();
}

async function deleteAlertCard(event, card, data) {
    event.stopPropagation();
    card.classList.add("removing");
    try {
        if (data.event_id) {
            const res = await fetch(`/api/events/${data.event_id}`, {
                method: "DELETE",
            });
            if (res.ok) {
                const body = await res.json();
                card.remove();
                applyStatsFromServer(body.stats);
                return;
            }
        }
        // Fallback nếu bản ghi không có event_id
        card.remove();
        if (alertCountsByType[data.event_type] > 0) {
            alertCountsByType[data.event_type]--;
        }
        totalAlerts = Math.max(0, totalAlerts - 1);
        renderAlertCounters();
    } catch (e) {
        console.error("Lỗi khi xoá cảnh báo:", e);
        card.classList.remove("removing");
    }
}

async function clearAlerts() {
    const visibleCount = Array.from(alertFeed.children).filter(
        (c) => c.style.display !== "none"
    ).length;
    if (visibleCount === 0 && totalAlerts === 0) return;

    try {
        const res = await fetch(
            `/api/events?event_type=${encodeURIComponent(activeFilter)}`,
            { method: "DELETE" }
        );
        if (res.ok) {
            const body = await res.json();
            for (const card of Array.from(alertFeed.children)) {
                if (
                    activeFilter === "ALL" ||
                    card.dataset.eventType === activeFilter
                ) {
                    card.remove();
                }
            }
            applyStatsFromServer(body.stats);
        }
    } catch (e) {
        console.error("Lỗi khi xoá danh sách cảnh báo:", e);
    }
}

function addAlertCard(data) {
    const card = document.createElement("div");
    card.className = `alert-card ${data.event_type}`;
    card.dataset.eventType = data.event_type;
    if (data.event_id) {
        card.dataset.eventId = data.event_id;
    }
    if (activeFilter !== "ALL" && data.event_type !== activeFilter) {
        card.style.display = "none";
    }

    let icon = "⚠️";
    if (data.event_type === "FALLING") icon = "🚨";
    if (data.event_type === "RUNNING") icon = "🏃";
    if (data.event_type === "FIGHTING") icon = "🥊";
    const vnTitleMap = { RUNNING: "CHẠY NHANH", FALLING: "TÉ NGÃ", FIGHTING: "ẨU ĐẢ / ĐÁNH NHAU" };
    const vnTitle = vnTitleMap[data.event_type] || data.event_type;

    card.innerHTML = `
        <img class="alert-thumb" src="${data.snapshot_url}" alt="Snapshot" onerror="this.onerror=null; this.style.visibility='hidden'">
        <div class="alert-info">
            <div class="alert-title">
                <span class="alert-badge-text">${icon} ${vnTitle} <small class="track-tag">#${data.track_id}</small></span>
                <div class="alert-actions">
                    <span class="conf-pill">${data.confidence}%</span>
                    <button type="button" class="alert-delete-btn" title="Xoá cảnh báo này" aria-label="Xoá cảnh báo">✕</button>
                </div>
            </div>
            <div class="alert-desc">${data.extra_info || ("Phát hiện hành vi " + vnTitle)}</div>
            <div class="alert-time">🕒 ${data.timestamp}</div>
        </div>
    `;

    const delBtn = card.querySelector(".alert-delete-btn");
    if (delBtn) {
        delBtn.addEventListener("click", (e) => deleteAlertCard(e, card, data));
    }

    // Bấm vào thẻ cảnh báo để xem ảnh snapshot phóng to
    card.style.cursor = "pointer";
    card.title = "Bấm để xem ảnh sự kiện";
    card.addEventListener("click", () => openSnapshotModal(data, icon));

    alertFeed.insertBefore(card, alertFeed.firstChild);
    updateEmptyState();
}

// Hộp thoại xem lại ảnh snapshot của cảnh báo
function openSnapshotModal(data, icon) {
    const overlay = document.createElement("div");
    overlay.className = "snapshot-modal";
    overlay.innerHTML = `
        <div class="snapshot-modal-box">
            <img src="${data.snapshot_url}" alt="Snapshot" onerror="this.onerror=null; this.style.visibility='hidden'">
            <div class="snapshot-modal-info">
                <b>${icon} ${data.event_type}</b> (Track #${data.track_id}) | ${data.timestamp}
                <div>${data.extra_info || ""}</div>
            </div>
        </div>
    `;
    const close = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    overlay.addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
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

        alertFeed.innerHTML = "";
        // API trả về mới nhất trước, còn addAlertCard chèn lên đầu nên duyệt ngược
        for (const ev of events.slice().reverse()) {
            addAlertCard(ev);
        }

        applyStatsFromServer(stats);
    } catch (e) {
        console.error("Lỗi khi tải lịch sử cảnh báo:", e);
        updateEmptyState();
    }
}

// Không dùng window.onload: thẻ <img> MJPEG stream không bao giờ tải xong nên sự kiện load không bao giờ xảy ra
loadAlertHistory();

let bannerTimer = null;
function showLiveBanner(data) {
    const banner = document.getElementById("liveAlertBanner");
    if (!banner) return;
    const vnTitleMap = { RUNNING: "CHẠY NHANH", FALLING: "TÉ NGÃ KHẨN CẤP", FIGHTING: "ẨU ĐẢ / XÔ XÁT" };
    const title = vnTitleMap[data.event_type] || data.event_type;
    banner.innerHTML = `🚨 CẢNH BÁO ${title}: ${data.extra_info || ("Đối tượng ID #" + data.track_id)}`;
    banner.style.display = "block";
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => { banner.style.display = "none"; }, 4000);
}
function playBeepSound(eventType) {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const isUrgent = (eventType === "FIGHTING" || eventType === "FALLING");
        const freqs = isUrgent ? [960, 720, 960] : [880];
        freqs.forEach((freq, idx) => {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = isUrgent ? "sawtooth" : "sine";
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            const t0 = audioCtx.currentTime + idx * 0.14;
            osc.frequency.setValueAtTime(freq, t0);
            gain.gain.setValueAtTime(0.22, t0);
            gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.13);
            osc.start(t0);
            osc.stop(t0 + 0.13);
        });
    } catch (e) {
        // Trình duyệt có thể chặn autoplay
    }
}

// Hô hào cảnh báo trực tiếp bằng giọng nói tiếng Việt như người giám sát thật (Web Speech API)
let lastSpeechTime = 0;
function speakAlertCallout(data) {
    try {
        if (!("speechSynthesis" in window)) return;
        const now = Date.now();
        // Tránh đọc chồng chéo quá dày (cách nhau tối thiểu 2.5s, ưu tiên FIGHTING, FALLING)
        const isCritical = (data.event_type === "FIGHTING" || data.event_type === "FALLING");
        if (!isCritical && (now - lastSpeechTime < 3500)) return;
        if (isCritical && (now - lastSpeechTime < 2200)) return;

        let text = "";
        if (data.event_type === "FIGHTING") {
            text = data.extra_info
                ? `Cảnh báo khẩn cấp! ${data.extra_info.replace(/\(V=.*?\)/g, "").replace(/#/g, "số ")}`
                : "Cảnh báo khẩn cấp! Phát hiện có người đang đánh nhau, xô xát mạnh!";
        } else if (data.event_type === "FALLING") {
            text = `Cảnh báo khẩn cấp! Phát hiện người số ${data.track_id} bị ngã nằm trên sàn!`;
        } else if (data.event_type === "RUNNING") {
            text = `Chú ý! Phát hiện đối tượng số ${data.track_id} đang chạy bứt tốc!`;
        }
        if (!text) return;

        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = "vi-VN";
        utter.rate = 1.15;
        utter.pitch = isCritical ? 1.1 : 1.0;
        const voices = window.speechSynthesis.getVoices();
        const viVoice = voices.find(v => v.lang && v.lang.toLowerCase().includes("vi"));
        if (viVoice) utter.voice = viVoice;
        window.speechSynthesis.speak(utter);
        lastSpeechTime = now;
    } catch (e) {
        console.warn("Speech synthesis error:", e);
    }
}

// -------------------------------------------------------------
// CHUYỂN ĐỔI NGUỒN VIDEO & UPLOAD VIDEO NGAY TRÊN GIAO DIỆN
// -------------------------------------------------------------
function selectActiveSource(sourceValue) {
    document.getElementById("videoSelect").value = sourceValue;
    for (const button of document.querySelectorAll(".demo-source")) {
        button.setAttribute("aria-pressed", String(button.dataset.source === sourceValue));
    }
}

async function onSelectVideoSource(sourceValue, sourceLabel) {
    try {
        const response = await fetch("/api/change_source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: sourceValue })
        });
        if (response.ok) {
            const select = document.getElementById("videoSelect");
            if (sourceLabel && !Array.from(select.options).some(option => option.value === sourceValue)) {
                select.add(new Option(sourceLabel, sourceValue));
            }
            console.log("[INFO] Đã chuyển nguồn video sang:", sourceValue);
            selectActiveSource(sourceValue);
            clearTimeout(bannerTimer);
            document.getElementById("liveAlertBanner").style.display = "none";
            // Tải lại thẻ img video_feed để làm mới luồng stream
            refreshVideoStream();
            return true;
        } else {
            const err = await response.json().catch(() => ({}));
            if (sourceLabel) return false;
            alert("Không chuyển được nguồn video: " + (err.message || response.status));
        }
    } catch (e) {
        console.error("Lỗi khi chuyển nguồn video:", e);
    }
    return false;
}

document.getElementById("phoneCameraForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.getElementById("phoneCameraUrl");
    const status = document.getElementById("phoneCameraStatus");
    const button = event.currentTarget.querySelector('button[type="submit"]');
    let raw = input.value.trim();
    if (!raw) return;
    if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(raw)) {
        raw = `http://${raw}`;
    }
    let url;
    try {
        url = new URL(raw);
    } catch {
        status.textContent = "Địa chỉ luồng video không hợp lệ.";
        return;
    }
    if (!["http:", "https:", "rtsp:"].includes(url.protocol) || !url.hostname) {
        status.textContent = "Dùng địa chỉ luồng bắt đầu bằng http://, https:// hoặc rtsp://.";
        return;
    }
    // Tự động bổ sung cổng 4747 (DroidCam) nếu chỉ nhập IP và bổ sung /video nếu chưa có đường dẫn
    if ((url.protocol === "http:" || url.protocol === "https:") && (!url.pathname || url.pathname === "/")) {
        if (!url.port) url.port = "4747";
        url.pathname = "/video";
    }
    input.value = url.href;
    button.disabled = true;
    status.textContent = "⏳ Đang kết nối camera…";
    try {
        const connected = await onSelectVideoSource(url.href, "Camera điện thoại");
        status.textContent = connected
            ? "✅ Đã kết nối camera điện thoại."
            : "❌ Không mở được luồng. Kiểm tra camera đã phát, địa chỉ luồng và mạng Wi-Fi.";
    } finally {
        button.disabled = false;
    }
});

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
            selectActiveSource(data.video_path);
            
            refreshVideoStream();
        } else {
            const err = await res.json().catch(() => ({}));
            alert("Tải video thất bại: " + (err.message || res.status));
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

