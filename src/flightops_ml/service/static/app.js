/**
 * AeroPredict IO — Operations Console & Model Diagnostics
 * Version: 2.0.0
 */

document.addEventListener("DOMContentLoaded", async () => {
  let hubsData = [];
  let map = null;
  let hubMarkers = {};
  let routePolyline = null;
  let isSubmitting = false;

  // 1. Navigation Tabs
  const navLinks = document.querySelectorAll(".nav-link");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const targetId = link.getAttribute("data-tab");

      navLinks.forEach((l) => {
        l.classList.remove("active");
        l.setAttribute("aria-selected", "false");
      });
      tabPanes.forEach((p) => p.classList.remove("active"));

      link.classList.add("active");
      link.setAttribute("aria-selected", "true");

      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (targetId === "tab-operations" && map) {
        setTimeout(() => map.invalidateSize(), 150);
      }
    });
  });

  // 2. Fetch Hub Infrastructure Data
  try {
    const res = await fetch("/static/hubs.json");
    if (!res.ok) throw new Error(`HTTP status ${res.status}`);
    hubsData = await res.json();
    populateHubDropdowns(hubsData);
    initMap(hubsData);
  } catch (err) {
    console.error("Failed to load hubs metadata:", err);
  }

  // 3. Populate Airport Dropdowns
  function populateHubDropdowns(hubs) {
    const originSelect = document.getElementById("origin-select");
    const destSelect = document.getElementById("dest-select");

    if (!originSelect || !destSelect) return;

    originSelect.innerHTML = "";
    destSelect.innerHTML = "";

    hubs.forEach((hub) => {
      const optOrigin = document.createElement("option");
      optOrigin.value = hub.iata;
      optOrigin.textContent = `${hub.iata} — ${hub.name}`;
      if (hub.iata === "ORD") optOrigin.selected = true;
      originSelect.appendChild(optOrigin);

      const optDest = document.createElement("option");
      optDest.value = hub.iata;
      optDest.textContent = `${hub.iata} — ${hub.name}`;
      if (hub.iata === "LGA") optDest.selected = true;
      destSelect.appendChild(optDest);
    });

    originSelect.addEventListener("change", () => {
      markCustomScenario();
      updateRoute();
    });

    destSelect.addEventListener("change", () => {
      markCustomScenario();
      updateRoute();
    });
  }

  // 4. Initialize Leaflet Tactical Map
  function initMap(hubs) {
    const mapElement = document.getElementById("flight-map");
    if (!mapElement) return;

    map = L.map("flight-map", {
      zoomControl: true,
      attributionControl: false,
    }).setView([39.2, -98.2], 4);

    L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        maxZoom: 10,
        minZoom: 3,
      }
    ).addTo(map);

    hubs.forEach((hub) => {
      let color = "#10b981"; // Emerald (< 20%)
      if (hub.delay_rate_pct >= 25.0) {
        color = "#ef4444"; // Rose (> 25%)
      } else if (hub.delay_rate_pct >= 20.0) {
        color = "#f59e0b"; // Amber (20-25%)
      }

      const radius = Math.min(13, Math.max(5.5, Math.sqrt(hub.flight_count) / 48));

      const marker = L.circleMarker([hub.lat, hub.lon], {
        radius: radius,
        fillColor: color,
        color: "#ffffff",
        weight: 1.2,
        opacity: 0.9,
        fillOpacity: 0.8,
      }).addTo(map);

      const popupHtml = `
        <div class="hub-popup-corp">
          <div class="iata-code">${hub.iata}</div>
          <div class="name">${hub.name}</div>
          <div class="stat-row">
            <span>Annual Movements</span>
            <strong>${hub.flight_count.toLocaleString()}</strong>
          </div>
          <div class="stat-row">
            <span>Baseline Delay Frequency</span>
            <strong style="color: ${color}">${hub.delay_rate_pct}%</strong>
          </div>
          <div class="action-buttons">
            <button onclick="selectAirport('${hub.iata}', 'origin')">Set Origin</button>
            <button onclick="selectAirport('${hub.iata}', 'dest')">Set Dest</button>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml);
      hubMarkers[hub.iata] = marker;
    });

    updateRoute();

    window.addEventListener("resize", () => {
      if (map) map.invalidateSize();
    });
  }

  window.selectAirport = (iata, type) => {
    if (type === "origin") {
      const el = document.getElementById("origin-select");
      if (el) el.value = iata;
    } else {
      const el = document.getElementById("dest-select");
      if (el) el.value = iata;
    }
    markCustomScenario();
    updateRoute();
    if (map) map.closePopup();
  };

  function updateRoute() {
    if (!map || hubsData.length === 0) return;

    const originSelect = document.getElementById("origin-select");
    const destSelect = document.getElementById("dest-select");
    if (!originSelect || !destSelect) return;

    const originIata = originSelect.value;
    const destIata = destSelect.value;

    const originHub = hubsData.find((h) => h.iata === originIata);
    const destHub = hubsData.find((h) => h.iata === destIata);

    if (routePolyline) {
      map.removeLayer(routePolyline);
      routePolyline = null;
    }

    if (originHub && destHub && originIata !== destIata) {
      const latlngs = [
        [originHub.lat, originHub.lon],
        [destHub.lat, destHub.lon],
      ];

      routePolyline = L.polyline(latlngs, {
        color: "#0284c7",
        weight: 3.0,
        opacity: 0.9,
        dashArray: "5, 6",
      }).addTo(map);

      map.fitBounds(routePolyline.getBounds(), { padding: [45, 45] });
    }
  }

  // 5. Form Sliders & Accessibility Sync
  const depHourInput = document.getElementById("dep-hour");
  const depHourVal = document.getElementById("dep-hour-val");
  if (depHourInput && depHourVal) {
    depHourInput.addEventListener("input", (e) => {
      const hour = parseInt(e.target.value);
      const text = `${String(hour).padStart(2, "0")}:00 (${String(hour).padStart(2, "0")}:00 LT)`;
      depHourVal.textContent = text;
      depHourInput.setAttribute("aria-valuenow", hour);
      depHourInput.setAttribute("aria-valuetext", text);
      markCustomScenario();
    });
  }

  const queueInput = document.getElementById("queue-input");
  const queueVal = document.getElementById("queue-val");
  if (queueInput && queueVal) {
    queueInput.addEventListener("input", (e) => {
      const val = e.target.value;
      const text = `${val} departures/hr`;
      queueVal.textContent = text;
      queueInput.setAttribute("aria-valuenow", val);
      queueInput.setAttribute("aria-valuetext", text);
      markCustomScenario();
    });
  }

  const windInput = document.getElementById("wind-input");
  const windVal = document.getElementById("wind-val");
  if (windInput && windVal) {
    windInput.addEventListener("input", (e) => {
      const val = e.target.value;
      const text = `${val} kt`;
      windVal.textContent = text;
      windInput.setAttribute("aria-valuenow", val);
      windInput.setAttribute("aria-valuetext", text);
      markCustomScenario();
    });
  }

  const visInput = document.getElementById("vis-input");
  const visVal = document.getElementById("vis-val");
  if (visInput && visVal) {
    visInput.addEventListener("input", (e) => {
      const val = e.target.value;
      const text = `${parseFloat(val).toFixed(1)} miles`;
      visVal.textContent = text;
      visInput.setAttribute("aria-valuenow", val);
      visInput.setAttribute("aria-valuetext", text);
      markCustomScenario();
    });
  }

  // Carrier and Day select change listeners
  document.getElementById("carrier-select")?.addEventListener("change", markCustomScenario);
  document.getElementById("day-select")?.addEventListener("change", markCustomScenario);

  // 6. METAR Weather Condition Selector
  const weatherBtns = document.querySelectorAll(".btn-metar");
  weatherBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      weatherBtns.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-checked", "true");

      const type = btn.getAttribute("data-weather");
      applyWeatherPreset(type);
      markCustomScenario();
    });
  });

  function applyWeatherPreset(type) {
    if (type === "clear") {
      windInput.value = 6;
      windVal.textContent = "6 kt";
      visInput.value = 10;
      visVal.textContent = "10.0 miles";
      window.currentWeatherFlags = { precip: 0, snow: 0, storm: 0, fog: 0, temp: 68, dew: 46 };
    } else if (type === "rain") {
      windInput.value = 16;
      windVal.textContent = "16 kt";
      visInput.value = 4;
      visVal.textContent = "4.0 miles";
      window.currentWeatherFlags = { precip: 1, snow: 0, storm: 0, fog: 0, temp: 54, dew: 52 };
    } else if (type === "storm") {
      windInput.value = 28;
      windVal.textContent = "28 kt";
      visInput.value = 2;
      visVal.textContent = "2.0 miles";
      window.currentWeatherFlags = { precip: 1, snow: 0, storm: 1, fog: 0, temp: 72, dew: 70 };
    } else if (type === "snow") {
      windInput.value = 22;
      windVal.textContent = "22 kt";
      visInput.value = 1.5;
      visVal.textContent = "1.5 miles";
      window.currentWeatherFlags = { precip: 1, snow: 1, storm: 0, fog: 0, temp: 28, dew: 26 };
    } else if (type === "fog") {
      windInput.value = 4;
      windVal.textContent = "4 kt";
      visInput.value = 0.5;
      visVal.textContent = "0.5 miles";
      window.currentWeatherFlags = { precip: 0, snow: 0, storm: 0, fog: 1, temp: 48, dew: 48 };
    }

    windInput.setAttribute("aria-valuenow", windInput.value);
    visInput.setAttribute("aria-valuenow", visInput.value);
  }

  // Default weather flags: Snow scenario
  window.currentWeatherFlags = { precip: 1, snow: 1, storm: 0, fog: 0, temp: 28, dew: 26 };

  // 7. Scenario Presets Handling
  const scenarioBtns = document.querySelectorAll(".btn-scenario");
  const activeScenarioBadge = document.getElementById("active-scenario-badge");

  function markCustomScenario() {
    scenarioBtns.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-pressed", "false");
    });
    if (activeScenarioBadge) {
      activeScenarioBadge.textContent = "Custom Configuration";
    }
  }

  scenarioBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      scenarioBtns.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");

      const preset = btn.getAttribute("data-preset");
      if (preset === "ord_storm") {
        document.getElementById("carrier-select").value = "UA";
        document.getElementById("origin-select").value = "ORD";
        document.getElementById("dest-select").value = "LGA";
        depHourInput.value = 19;
        depHourInput.dispatchEvent(new Event("input"));
        queueInput.value = 45;
        queueInput.dispatchEvent(new Event("input"));
        document.querySelector('.btn-metar[data-weather="snow"]')?.click();
        if (activeScenarioBadge) activeScenarioBadge.textContent = "ORD Winter Snow";
      } else if (preset === "atl_morning") {
        document.getElementById("carrier-select").value = "DL";
        document.getElementById("origin-select").value = "ATL";
        document.getElementById("dest-select").value = "MCO";
        depHourInput.value = 7;
        depHourInput.dispatchEvent(new Event("input"));
        queueInput.value = 14;
        queueInput.dispatchEvent(new Event("input"));
        document.querySelector('.btn-metar[data-weather="clear"]')?.click();
        if (activeScenarioBadge) activeScenarioBadge.textContent = "ATL Nominal VFR";
      } else if (preset === "den_blizzard") {
        document.getElementById("carrier-select").value = "WN";
        document.getElementById("origin-select").value = "DEN";
        document.getElementById("dest-select").value = "SFO";
        depHourInput.value = 17;
        depHourInput.dispatchEvent(new Event("input"));
        queueInput.value = 38;
        queueInput.dispatchEvent(new Event("input"));
        document.querySelector('.btn-metar[data-weather="snow"]')?.click();
        if (activeScenarioBadge) activeScenarioBadge.textContent = "DEN Blizzard Ops";
      } else if (preset === "dfw_evening") {
        document.getElementById("carrier-select").value = "AA";
        document.getElementById("origin-select").value = "DFW";
        document.getElementById("dest-select").value = "LAX";
        depHourInput.value = 18;
        depHourInput.dispatchEvent(new Event("input"));
        queueInput.value = 42;
        queueInput.dispatchEvent(new Event("input"));
        document.querySelector('.btn-metar[data-weather="storm"]')?.click();
        if (activeScenarioBadge) activeScenarioBadge.textContent = "DFW Peak Bank";
      }
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
      updateRoute();
    });
  });

  // 8. Flight Risk Inference Submission
  const form = document.getElementById("flight-simulator-form");
  const submitBtn = document.getElementById("btn-submit-predict");
  const errorBanner = document.getElementById("form-error-banner");
  const errorMsg = document.getElementById("form-error-msg");
  const evalPlaceholder = document.getElementById("eval-placeholder");
  const evalActiveContainer = document.getElementById("eval-active-container");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      if (isSubmitting) return;
      isSubmitting = true;

      if (errorBanner) errorBanner.style.display = "none";
      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <svg viewBox="0 0 24 24" class="spin" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
        <span>Computing T-120m TreeSHAP Inference...</span>
      `;

      const carrier = document.getElementById("carrier-select").value;
      const origin = document.getElementById("origin-select").value;
      const dest = document.getElementById("dest-select").value;
      const depHour = parseInt(depHourInput.value);
      const dayOfWeek = parseInt(document.getElementById("day-select").value);
      const month = parseInt(document.getElementById("month-select").value);
      const queue = parseFloat(queueInput.value);
      const windSpeed = parseFloat(windInput.value);
      const visibility = parseFloat(visInput.value);

      const originHub = hubsData.find((h) => h.iata === origin);
      const destHub = hubsData.find((h) => h.iata === dest);

      let originLat = originHub ? originHub.lat : null;
      let originLon = originHub ? originHub.lon : null;
      let destLat = destHub ? destHub.lat : null;
      let destLon = destHub ? destHub.lon : null;

      // Omit/pass null for distance and duration so backend resolves authoritative historical route
      // profiles (preserving directional block times). Unprofiled routes fall back automatically.
      const payload = {
        op_unique_carrier: carrier,
        origin: origin,
        dest: dest,
        dep_hour: depHour,
        day_of_week: dayOfWeek,
        month: month,
        distance: null,
        crs_elapsed_time_clean: null,
        origin_latitude: originLat,
        origin_longitude: originLon,
        dest_latitude: destLat,
        dest_longitude: destLon,
        weather_temp_f: window.currentWeatherFlags.temp,
        weather_dew_point_f: window.currentWeatherFlags.dew,
        weather_wind_speed_kt: windSpeed,
        weather_wind_gust_kt: windSpeed > 15 ? windSpeed + 12 : 0.0,
        weather_visibility_sm: visibility,
        weather_altimeter_in_hg: 29.92,
        weather_has_precip: window.currentWeatherFlags.precip,
        weather_has_thunderstorm: window.currentWeatherFlags.storm,
        weather_has_snow: window.currentWeatherFlags.snow,
        weather_has_fog_haze: window.currentWeatherFlags.fog,
        origin_hourly_flights: queue,
        carrier_hub_daily_flights: 120.0,
        hub_daily_total_flights: 340.0,
      };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`Inference endpoint returned HTTP status ${response.status}`);
        }

        const data = await response.json();
        renderInferenceDiagnostics(data);
      } catch (err) {
        console.error("Inference execution failed:", err);
        if (errorBanner && errorMsg) {
          errorMsg.textContent = `Inference failed: ${err.message}. Selected parameters preserved.`;
          errorBanner.style.display = "flex";
        }
      } finally {
        isSubmitting = false;
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
          <svg viewBox="0 0 24 24" aria-hidden="true"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          <span>Evaluate Flight Delay Risk (T-120m)</span>
        `;
      }
    });
  }

  // 9. Render Inference Diagnostics & TreeSHAP Table
  function renderInferenceDiagnostics(data) {
    if (evalPlaceholder) evalPlaceholder.style.display = "none";
    if (evalActiveContainer) evalActiveContainer.style.display = "block";

    const resultCard = document.getElementById("prediction-result");
    if (resultCard) {
      resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    const banner = document.getElementById("result-banner");

    const bannerTitle = document.getElementById("banner-title");
    const bannerSubtitle = document.getElementById("banner-subtitle");
    const iconBox = document.getElementById("status-icon-box");
    const gaugeProb = document.getElementById("gauge-prob-val");
    const latencyVal = document.getElementById("latency-val");

    const probPct = (data.predicted_delay_probability * 100).toFixed(1);
    gaugeProb.textContent = `${probPct}%`;
    latencyVal.textContent = `${data.inference_latency_ms} ms`;

    if (data.alert_triggered) {
      banner.className = "result-status-header status-alert";
      iconBox.className = "status-icon-box icon-alert";
      iconBox.innerHTML = `<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
      bannerTitle.textContent = `DISPATCH ALERT: ELEVATED DEPARTURE DELAY RISK (${data.risk_tier})`;
      bannerSubtitle.textContent = `Predicted probability exceeds the calibrated operational threshold (0.21). Recommend proactive gate turnaround buffer and tail swap contingency.`;
      gaugeProb.style.color = "var(--status-rose)";
    } else {
      banner.className = "result-status-header status-cleared";
      iconBox.className = "status-icon-box icon-cleared";
      iconBox.innerHTML = `<svg viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
      bannerTitle.textContent = `DISPATCH CLEARANCE: NOMINAL SCHEDULE RECOVERY EXPECTED (${data.risk_tier})`;
      bannerSubtitle.textContent = `Predicted delay probability is below the operational threshold (0.21). Standard turnaround clearance approved.`;
      gaugeProb.style.color = "var(--status-emerald)";
    }

    const tableBody = document.getElementById("factors-table-body");
    tableBody.innerHTML = "";

    // Drivers UP
    data.top_delay_reasons.forEach((reason) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="feature-name">${escapeHtml(reason.display_name)}</td>
        <td class="observed-val">${escapeHtml(formatObservedValue(reason.feature_value))}</td>
        <td class="impact-val impact-pos">+${(reason.shap_impact * 100).toFixed(1)}%</td>
        <td>${escapeHtml(reason.description)}</td>
      `;
      tableBody.appendChild(tr);
    });

    // Drivers DOWN
    data.top_mitigating_factors.forEach((mitigator) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="feature-name">${escapeHtml(mitigator.display_name)}</td>
        <td class="observed-val">${escapeHtml(formatObservedValue(mitigator.feature_value))}</td>
        <td class="impact-val impact-neg">${(mitigator.shap_impact * 100).toFixed(1)}%</td>
        <td>${escapeHtml(mitigator.description)}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  function formatObservedValue(val) {
    if (typeof val === "number") {
      return Number.isInteger(val) ? val.toString() : val.toFixed(2);
    }
    return String(val);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // 10. Inspection Modal for High-Resolution Figures
  const modal = document.getElementById("img-modal");
  const modalImg = document.getElementById("modal-img");
  const modalClose = document.getElementById("modal-close");

  document.querySelectorAll(".chart-preview-frame").forEach((frame) => {
    frame.addEventListener("click", () => {
      const img = frame.querySelector("img");
      if (img && modal && modalImg) {
        modalImg.src = img.src;
        modal.classList.add("active");
      }
    });
  });

  if (modalClose) {
    modalClose.addEventListener("click", () => {
      modal.classList.remove("active");
    });
  }

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.classList.remove("active");
      }
    });
  }
});
