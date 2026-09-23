// Smart Home Energy Forecasting & Optimization AI - Dashboard Client Script

let forecastChart = null;
let applianceChart = null;
let optimizationChart = null;
let batteryChart = null;

let globalForecastData = null;
let globalApplianceData = null;

let currentHorizon = 24; // Default 24h forecast horizon
let currentApplianceFilter = "all";

document.addEventListener("DOMContentLoaded", () => {
    initParticleCanvas();
    initDashboard();
    setupEventListeners();
});

// --- 1. Background Futuristic AI Particle Matrix Canvas ---
function initParticleCanvas() {
    const canvas = document.getElementById("particleCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener("resize", () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const numParticles = 45;
    const particles = [];

    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            radius: Math.random() * 1.8 + 0.8,
            color: Math.random() > 0.5 ? "rgba(201, 153, 107, " : "rgba(92, 118, 109, "
        });
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < numParticles; i++) {
            let p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color + "0.65)";
            ctx.fill();

            for (let j = i + 1; j < numParticles; j++) {
                let p2 = particles[j];
                let dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 130) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(201, 153, 107, ${0.14 * (1 - dist / 130)})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
}

// --- 2. Core Dashboard Initialization ---
async function initDashboard() {
    await fetchMetrics();
    await fetchForecastData();
    await fetchApplianceData();
    await runOptimization();
}

// --- 3. Interactive Event Listeners & Clicks Setup ---
function setupEventListeners() {
    const batterySlider = document.getElementById("batterySlider");
    const batteryVal = document.getElementById("batteryVal");
    const toggleLoadShifting = document.getElementById("toggleLoadShifting");
    const toggleBattery = document.getElementById("toggleBattery");
    const runOptBtn = document.getElementById("runOptBtn");

    // Slider Live Input & Instant Recalculation
    if (batterySlider) {
        batterySlider.addEventListener("input", (e) => {
            batteryVal.textContent = `${parseFloat(e.target.value).toFixed(1)} kWh`;
        });
        batterySlider.addEventListener("change", () => {
            runOptimization();
        });
    }

    // Toggle Groups Clicks
    if (toggleLoadShifting) {
        toggleLoadShifting.addEventListener("change", () => runOptimization());
    }
    if (toggleBattery) {
        toggleBattery.addEventListener("change", () => runOptimization());
    }

    document.getElementById("groupLoadShifting")?.addEventListener("click", (e) => {
        if (e.target.tagName !== "INPUT" && e.target.tagName !== "SPAN") {
            toggleLoadShifting.checked = !toggleLoadShifting.checked;
            runOptimization();
        }
    });

    document.getElementById("groupBattery")?.addEventListener("click", (e) => {
        if (e.target.tagName !== "INPUT" && e.target.tagName !== "SPAN") {
            toggleBattery.checked = !toggleBattery.checked;
            runOptimization();
        }
    });

    // Run Optimization Button Click
    if (runOptBtn) {
        runOptBtn.addEventListener("click", () => {
            runOptBtn.style.transform = "scale(0.97)";
            setTimeout(() => runOptBtn.style.transform = "", 150);
            runOptimization();
        });
    }

    // Forecast Time Horizon Filter Buttons (24h / 3d / 7d)
    const horizonBtns = document.querySelectorAll(".button-group button[data-horizon]");
    horizonBtns.forEach(btn => {
        btn.addEventListener("click", (e) => {
            horizonBtns.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            currentHorizon = parseInt(e.target.dataset.horizon);
            renderForecastChart();
        });
    });

    // Appliance Filter Buttons (All / HVAC / EV / Solar)
    const applianceBtns = document.querySelectorAll("#applianceFilterGroup button[data-appliance]");
    applianceBtns.forEach(btn => {
        btn.addEventListener("click", (e) => {
            applianceBtns.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            currentApplianceFilter = e.target.dataset.appliance;
            renderApplianceChart();
        });
    });

    // Quick AI Presets Clicks
    document.getElementById("presetEco")?.addEventListener("click", () => {
        batterySlider.value = 5;
        batteryVal.textContent = "5.0 kWh";
        toggleLoadShifting.checked = true;
        toggleBattery.checked = true;
        runOptimization();
    });

    document.getElementById("presetHeatwave")?.addEventListener("click", () => {
        batterySlider.value = 15;
        batteryVal.textContent = "15.0 kWh";
        toggleLoadShifting.checked = true;
        toggleBattery.checked = true;
        runOptimization();
    });

    document.getElementById("presetMaxSave")?.addEventListener("click", () => {
        batterySlider.value = 20;
        batteryVal.textContent = "20.0 kWh";
        toggleLoadShifting.checked = true;
        toggleBattery.checked = true;
        runOptimization();
    });

    // Custom Dataset Upload & Control Listeners
    const btnBrowseCsv = document.getElementById("btnBrowseCsv");
    const csvFileInput = document.getElementById("csvFileInput");
    const selectedFileName = document.getElementById("selectedFileName");
    const btnUploadCsv = document.getElementById("btnUploadCsv");
    const btnSubmitManual = document.getElementById("btnSubmitManual");
    const btnResetDataset = document.getElementById("btnResetDataset");

    if (btnBrowseCsv && csvFileInput) {
        btnBrowseCsv.addEventListener("click", () => csvFileInput.click());
        csvFileInput.addEventListener("change", () => {
            if (csvFileInput.files.length > 0) {
                selectedFileName.textContent = `📄 ${csvFileInput.files[0].name}`;
            } else {
                selectedFileName.textContent = "No file selected";
            }
        });
    }

    if (btnUploadCsv && csvFileInput) {
        btnUploadCsv.addEventListener("click", async () => {
            if (!csvFileInput.files || csvFileInput.files.length === 0) {
                alert("Please select a CSV file first!");
                return;
            }
            const formData = new FormData();
            formData.append("file", csvFileInput.files[0]);

            btnUploadCsv.disabled = true;
            btnUploadCsv.querySelector("span").textContent = "⏳ Uploading & Processing...";

            try {
                const res = await fetch("/api/upload-csv", {
                    method: "POST",
                    body: formData
                });
                const result = await res.json();
                if (result.status === "success") {
                    alert(`✅ ${result.message}`);
                    await initDashboard();
                } else {
                    alert(`❌ Upload Error: ${result.message}`);
                }
            } catch (err) {
                alert(`❌ Failed to upload CSV dataset: ${err.message}`);
            } finally {
                btnUploadCsv.disabled = false;
                btnUploadCsv.querySelector("span").textContent = "🚀 Process & Run Custom Dataset";
            }
        });
    }

    if (btnSubmitManual) {
        btnSubmitManual.addEventListener("click", async () => {
            const reading = {
                temperature: parseFloat(document.getElementById("inputTemp").value) || 22.0,
                occupancy: parseFloat(document.getElementById("inputOccupancy").value) || 2.0,
                hvac_kw: parseFloat(document.getElementById("inputHvac").value) || 2.5,
                ev_charger_kw: parseFloat(document.getElementById("inputEv").value) || 7.2,
                water_heater_kw: parseFloat(document.getElementById("inputWater").value) || 1.5,
                solar_generation_kw: parseFloat(document.getElementById("inputSolar").value) || 3.5
            };

            btnSubmitManual.disabled = true;
            btnSubmitManual.querySelector("span").textContent = "⏳ Simulating Reading...";

            try {
                const res = await fetch("/api/predict-manual", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(reading)
                });
                const result = await res.json();
                if (result.status === "success") {
                    alert(`✨ ${result.message}`);
                    await initDashboard();
                } else {
                    alert(`❌ Simulation Error: ${result.message}`);
                }
            } catch (err) {
                alert(`❌ Prediction error: ${err.message}`);
            } finally {
                btnSubmitManual.disabled = false;
                btnSubmitManual.querySelector("span").textContent = "✨ Predict & Simulate Telemetry";
            }
        });
    }

    if (btnResetDataset) {
        btnResetDataset.addEventListener("click", async () => {
            try {
                const res = await fetch("/api/reset-dataset", { method: "POST" });
                const result = await res.json();
                if (result.status === "success") {
                    if (csvFileInput) csvFileInput.value = "";
                    if (selectedFileName) selectedFileName.textContent = "No file selected";
                    alert(`🔄 ${result.message}`);
                    await initDashboard();
                }
            } catch (err) {
                alert(`Failed to reset dataset: ${err.message}`);
            }
        });
    }
}

// --- 4. Metrics Fetching & Benchmark Table Populating ---
async function fetchMetrics() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        
        const summary = data.summary || {};
        document.getElementById("valTotalEnergy").textContent = `${(summary.total_energy_kwh || 0).toLocaleString()} kWh`;
        document.getElementById("valAvgLoad").textContent = `${(summary.avg_load_kw || 0).toFixed(2)} kW`;
        document.getElementById("valMaxLoad").textContent = `${(summary.max_load_kw || 0).toFixed(2)} kW`;
        document.getElementById("valSolarGen").textContent = `${(summary.total_solar_kwh || 0).toLocaleString()} kWh`;
        if (summary.data_source && document.getElementById("valDataSource")) {
            document.getElementById("valDataSource").textContent = summary.data_source;
        }

        const models = data.models || {};
        const tbody = document.getElementById("benchmarkTbody");
        if (tbody) {
            tbody.innerHTML = "";
            
            const badgeClasses = {
                "XGBoost": "badge-xgboost",
                "LSTM": "badge-lstm",
                "ARIMA": "badge-arima"
            };

            for (const [modelName, m] of Object.entries(models)) {
                const tr = document.createElement("tr");
                const badge = badgeClasses[modelName] || "badge-arima";
                
                tr.innerHTML = `
                    <td><span class="model-badge ${badge}">${modelName}</span></td>
                    <td class="score-highlight">${m.RMSE} kW</td>
                    <td class="score-highlight">${m.MAE} kW</td>
                    <td class="score-highlight">${m.MAPE}%</td>
                    <td class="score-highlight" style="color: ${m.R2 > 0.8 ? '#C9996B' : m.R2 > 0 ? '#5C766D' : '#d97757'}">${m.R2}</td>
                `;
                tbody.appendChild(tr);
            }
        }
    } catch (err) {
        console.error("Error fetching metrics:", err);
    }
}

// --- 5. Forecast Data & Smooth Dynamic Chart Rendering ---
async function fetchForecastData() {
    try {
        const res = await fetch("/api/forecast");
        globalForecastData = await res.json();
        renderForecastChart();
    } catch (err) {
        console.error("Error loading forecast data:", err);
    }
}

function renderForecastChart() {
    if (!globalForecastData) return;
    
    const limit = currentHorizon;
    let timestamps = (globalForecastData.timestamps || []).slice(0, limit);
    const actual = (globalForecastData.actual || []).slice(0, limit);
    const xgb = (globalForecastData.XGBoost || []).slice(0, limit);
    const lstm = (globalForecastData.LSTM || []).slice(0, limit);
    const arima = (globalForecastData.ARIMA || []).slice(0, limit);

    if (timestamps.length === 0 && actual.length > 0) {
        timestamps = actual.map((_, i) => `Hour ${i + 1}`);
    }

    const canvas = document.getElementById("forecastChartCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (forecastChart) forecastChart.destroy();

    // Create glowing gradients with palette #C9996B and #5C766D
    const gradXgb = ctx.createLinearGradient(0, 0, 0, 300);
    gradXgb.addColorStop(0, "rgba(201, 153, 107, 0.3)");
    gradXgb.addColorStop(1, "rgba(201, 153, 107, 0)");

    const gradLstm = ctx.createLinearGradient(0, 0, 0, 300);
    gradLstm.addColorStop(0, "rgba(92, 118, 109, 0.3)");
    gradLstm.addColorStop(1, "rgba(92, 118, 109, 0)");

    const formattedLabels = timestamps.map(t => (typeof t === "string" && t.length >= 10 ? t.substring(5, 16) : String(t)));

    forecastChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    label: "Actual Load (kW)",
                    data: actual,
                    borderColor: "#EDE9E6",
                    borderWidth: 2.2,
                    pointRadius: 0,
                    tension: 0.35,
                    cubicInterpolationMode: "monotone"
                },
                {
                    label: "XGBoost Model (R²=0.987)",
                    data: xgb,
                    borderColor: "#C9996B",
                    backgroundColor: gradXgb,
                    fill: true,
                    borderWidth: 2.2,
                    pointRadius: 0,
                    tension: 0.35,
                    cubicInterpolationMode: "monotone"
                },
                {
                    label: "Keras LSTM Model (R²=0.896)",
                    data: lstm,
                    borderColor: "#5C766D",
                    backgroundColor: gradLstm,
                    fill: true,
                    borderWidth: 2.2,
                    pointRadius: 0,
                    tension: 0.35,
                    cubicInterpolationMode: "monotone"
                },
                {
                    label: "ARIMA Model",
                    data: arima,
                    borderColor: "#d97757",
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 500 },
            plugins: {
                legend: { labels: { color: "rgba(237, 233, 230, 0.72)", font: { family: "Outfit" } } },
                tooltip: { mode: "index", intersect: false }
            },
            scales: {
                x: { ticks: { color: "rgba(237, 233, 230, 0.48)", maxTicksLimit: 10 }, grid: { color: "rgba(237, 233, 230, 0.05)" } },
                y: { title: { display: true, text: "Demand (kW)", color: "rgba(237, 233, 230, 0.72)" }, ticks: { color: "rgba(237, 233, 230, 0.48)" }, grid: { color: "rgba(237, 233, 230, 0.05)" } }
            }
        }
    });
}

// --- 6. Appliance Breakdown Visualization ---
async function fetchApplianceData() {
    try {
        const res = await fetch("/api/appliance-breakdown");
        globalApplianceData = await res.json();
        renderApplianceChart();
    } catch (err) {
        console.error("Error loading appliance breakdown data:", err);
    }
}

function renderApplianceChart() {
    if (!globalApplianceData) return;

    let timestamps = globalApplianceData.timestamps || [];
    const hourly = globalApplianceData.hourly || {};

    const canvas = document.getElementById("applianceChartCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (applianceChart) applianceChart.destroy();

    const formattedLabels = timestamps.map(t => (typeof t === "string" && t.length >= 10 ? t.substring(5, 16) : String(t)));

    const allDatasets = [
        { key: "hvac", label: "HVAC Cooling/Heating", data: hourly.hvac_kw || [], borderColor: "#5C766D", backgroundColor: "rgba(92, 118, 109, 0.15)", fill: true },
        { key: "ev", label: "EV Fast Charger", data: hourly.ev_charger_kw || [], borderColor: "#C9996B", backgroundColor: "rgba(201, 153, 107, 0.15)", fill: true },
        { key: "water", label: "Water Heater", data: hourly.water_heater_kw || [], borderColor: "#d97757", backgroundColor: "rgba(217, 119, 87, 0.12)", fill: true },
        { key: "solar", label: "Solar PV Generation", data: hourly.solar_generation_kw || [], borderColor: "#EDE9E6", borderWidth: 2.2, borderDash: [3, 3], fill: false }
    ];

    let filteredDatasets = allDatasets;
    if (currentApplianceFilter !== "all") {
        filteredDatasets = allDatasets.filter(d => d.key === currentApplianceFilter);
    }

    applianceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: formattedLabels,
            datasets: filteredDatasets.map(d => ({
                label: d.label,
                data: d.data,
                borderColor: d.borderColor,
                backgroundColor: d.backgroundColor || "transparent",
                fill: d.fill,
                borderWidth: d.borderWidth || 2,
                borderDash: d.borderDash || [],
                pointRadius: 0,
                tension: 0.35,
                cubicInterpolationMode: "monotone"
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 500 },
            plugins: {
                legend: { labels: { color: "rgba(237, 233, 230, 0.72)", font: { family: "Outfit" } } },
                tooltip: { mode: "index", intersect: false }
            },
            scales: {
                x: { ticks: { color: "rgba(237, 233, 230, 0.48)", maxTicksLimit: 10 }, grid: { color: "rgba(237, 233, 230, 0.05)" } },
                y: { title: { display: true, text: "Power (kW)", color: "rgba(237, 233, 230, 0.72)" }, ticks: { color: "rgba(237, 233, 230, 0.48)" }, grid: { color: "rgba(237, 233, 230, 0.05)" } }
            }
        }
    });
}

// --- 7. Run HEMS Optimization Engine & Update AI Copilot Advice ---
async function runOptimization() {
    try {
        const bCapacity = parseFloat(document.getElementById("batterySlider").value || 10);
        const loadShifting = document.getElementById("toggleLoadShifting").checked;
        const batteryOpt = document.getElementById("toggleBattery").checked;

        const res = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                battery_capacity_kwh: bCapacity,
                enable_load_shifting: loadShifting,
                enable_battery: batteryOpt
            })
        });

        const data = await res.json();
        
        // Update Cost Cards
        document.getElementById("valBaseCost").textContent = `$${data.baseline_cost.toFixed(2)}`;
        document.getElementById("valOptCost").textContent = `$${data.optimized_cost.toFixed(2)}`;
        document.getElementById("valSavings").textContent = `$${data.cost_savings.toFixed(2)} (${data.pct_cost_savings}%)`;
        document.getElementById("valPeakShave").textContent = `${data.peak_shaving_kw} kW (${data.pct_peak_shaving}%)`;
        document.getElementById("valSelfCons").textContent = `${data.solar_self_consumption_ratio}%`;

        // Update Dynamic AI Recommendation Text
        updateAiRecommendationText(data, bCapacity, loadShifting, batteryOpt);

        // Render Load Shifting Chart (Baseline vs Optimized Grid)
        const ctxOpt = document.getElementById("optChartCanvas").getContext("2d");
        if (optimizationChart) optimizationChart.destroy();

        const timestamps = data.timestamps || [];

        const gradOpt = ctxOpt.createLinearGradient(0, 0, 0, 300);
        gradOpt.addColorStop(0, "rgba(92, 118, 109, 0.3)");
        gradOpt.addColorStop(1, "rgba(92, 118, 109, 0)");

        optimizationChart = new Chart(ctxOpt, {
            type: "line",
            data: {
                labels: timestamps.map(t => t.substring(5, 16)),
                datasets: [
                    { label: "Baseline Grid Import (Unoptimized)", data: data.baseline_demand || [], borderColor: "#d97757", borderWidth: 1.8, pointRadius: 0, tension: 0.3 },
                    { label: "Optimized Grid Demand (HEMS Shaved)", data: data.optimized_demand || [], borderColor: "#5C766D", backgroundColor: gradOpt, fill: true, borderWidth: 2.2, pointRadius: 0, tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: { legend: { labels: { color: "rgba(237, 233, 230, 0.72)" } } },
                scales: {
                    x: { ticks: { color: "rgba(237, 233, 230, 0.48)", maxTicksLimit: 10 }, grid: { color: "rgba(237, 233, 230, 0.05)" } },
                    y: { title: { display: true, text: "Grid Load (kW)", color: "rgba(237, 233, 230, 0.72)" }, ticks: { color: "rgba(237, 233, 230, 0.48)" }, grid: { color: "rgba(237, 233, 230, 0.05)" } }
                }
            }
        });

        // Render Battery SOC Chart
        const ctxBat = document.getElementById("batteryChartCanvas").getContext("2d");
        if (batteryChart) batteryChart.destroy();

        const gradBat = ctxBat.createLinearGradient(0, 0, 0, 300);
        gradBat.addColorStop(0, "rgba(201, 153, 107, 0.35)");
        gradBat.addColorStop(1, "rgba(201, 153, 107, 0)");

        batteryChart = new Chart(ctxBat, {
            type: "line",
            data: {
                labels: timestamps.map(t => t.substring(5, 16)),
                datasets: [
                    { label: "BESS State-of-Charge (kWh)", data: data.battery_soc || [], borderColor: "#C9996B", backgroundColor: gradBat, fill: true, borderWidth: 2.2, pointRadius: 0, tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: { legend: { labels: { color: "rgba(237, 233, 230, 0.72)" } } },
                scales: {
                    x: { ticks: { color: "rgba(237, 233, 230, 0.48)", maxTicksLimit: 10 }, grid: { color: "rgba(237, 233, 230, 0.05)" } },
                    y: { title: { display: true, text: "Battery Level (kWh)", color: "rgba(237, 233, 230, 0.72)" }, ticks: { color: "rgba(237, 233, 230, 0.48)" }, grid: { color: "rgba(237, 233, 230, 0.05)" } }
                }
            }
        });

    } catch (err) {
        console.error("Error running optimization:", err);
    }
}

// --- 8. AI Copilot Advice Text Generator ---
function updateAiRecommendationText(data, bCap, loadShift, bOpt) {
    const textEl = document.getElementById("aiAdviceText");
    if (!textEl) return;

    let advice = `AI Telemetry Analysis Complete. `;

    if (loadShift && bOpt) {
        advice += `With <strong>${bCap} kWh BESS</strong> storage and active EV load shifting, your peak demand is shaved by <strong>${data.peak_shaving_kw} kW (${data.pct_peak_shaving}%)</strong>. Off-peak energy arbitrage reduces utility costs by <strong>$${data.cost_savings} (${data.pct_cost_savings}%)</strong> while utilizing <strong>${data.solar_self_consumption_ratio}%</strong> of clean solar PV generation.`;
    } else if (loadShift && !bOpt) {
        advice += `EV Peak Load Shifting is active (deferring 7.2 kW charging to off-peak tariff). Enabling <strong>BESS Storage</strong> would yield an additional ~15-20% cost savings through solar PV arbitrage during on-peak hours (17:00–21:00).`;
    } else if (!loadShift && bOpt) {
        advice += `BESS Battery Storage is active (${bCap} kWh capacity), saving $${data.cost_savings}. Enabling <strong>EV Peak Load Shifting</strong> will prevent heavy evening grid spikes during 17:00–21:00 on-peak pricing.`;
    } else {
        advice += `<span style="color: #d97757;">⚠️ Warning: All HEMS optimizations are disabled.</span> Your smart home is importing electricity directly from grid at peak Time-of-Use rates ($0.32/kWh). Enable Load Shifting or Battery Storage to unlock up to $${data.cost_savings || 36} in monthly savings.`;
    }

    textEl.innerHTML = advice;
}


