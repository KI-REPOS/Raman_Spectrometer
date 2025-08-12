# Raman Spectrometer GUI (PySide6)

## 📌 Overview
This is a **PySide6-based desktop application** for operating a **CCD-based Raman Spectrometer** (tested with TCD1304 + STM32F401RE).

It provides:
- Real-time spectrum acquisition
- Wavelength calibration
- Raman shift calculation
- Interactive plotting
- Data/image saving
- File loading for review

---

<details>
<summary>⚙️ How It Works</summary>

### 1. Hardware Communication
- Connects to STM32 over **USB Serial (115200 baud)**.
- Sends CCD parameters: SH period, ICG period, mode, averages.
- Receives **3694 pixels × 16-bit** intensity data.

### 2. Data Processing
- Inverts intensity to show blocked-light dips as peaks.
- **Pixel → Wavelength** conversion via quadratic fit:
```bash
λ = a·p² + b·p + c
```
- **Raman Shift** calculation:
where λ₀ = excitation wavelength.

### 3. Interactive Graphs
- Pixel vs Intensity
- Wavelength vs Intensity (calibrated)
- Raman Shift vs Intensity (calibrated)
- Zoom, pan, reset, theme toggle
- Real-time X/Y cursor tracking

### 4. Calibration
- Add pixel–wavelength pairs manually.
- Fits quadratic curve for mapping.
- Shows error estimation & warnings.

### 5. File Operations
- **Upload** saved spectra (with/without wavelength/Raman data).
- **Save Data** (CSV/TXT with metadata & calibration).
- **Save Image** (PNG/SVG export).
</details>

---

<details>
<summary>🖥️ Application Interface</summary>

1. **Acquisition Control Panel**
 - Connect / Disconnect spectrometer
 - Start / Stop acquisition
 - Mode: Single / Continuous
 - Set averaging count

2. **CCD Settings**
 - SH Period & ICG Period inputs
 - Integration time calculator

3. **Graph Tabs**
 - Pixel vs Intensity
 - Wavelength vs Intensity
 - Raman Shift vs Intensity

4. **Bottom Controls**
 - Upload file
 - Select excitation wavelength
 - Theme toggle
 - Calibrate wavelength
 - Reset zoom
 - Save data
 - Save image
</details>

---

<details>
<summary>🚀 Installation</summary>

### Requirements
- Python 3.9+
- Install dependencies:
```bash
pip install PySide6 pyserial numpy matplotlib scipy
