import sys
import serial
import serial.tools.list_ports
import numpy as np
import csv
import time
import os
from datetime import datetime
from scipy.optimize import curve_fit

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QGroupBox, QTabWidget, QFileDialog, QMessageBox,
    QStatusBar, QCheckBox, QDialog
)
from PySide6.QtCore import QTimer, Qt, Signal, QObject, QThread
from PySide6.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

# Constants
MCLK = 2_000_000  # 2 MHz master clock
DEFAULT_SH_PERIOD = 200
DEFAULT_ICG_PERIOD = 100_000
DEFAULT_AVG = 1
DEFAULT_MODE = 0
EXCITATION_WAVELENGTHS = [532, 785, 1064]  # Common laser wavelengths
THEMES = ['light', 'dark']

class InteractiveGraph(FigureCanvas):
    cursor_moved = Signal(float, float)  # Signal to emit cursor position
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.data_line = None
        self.annotations = []
        self.current_theme = 'light'
        self.zoom_factor = 1.5
        self.pan_start = None
        self.setup_interactivity()

    def setup_interactivity(self):
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def plot_data(self, x, y, xlabel, ylabel, title):
        self.ax.clear()
        # Invert the y-axis data to show dips when light is blocked
        inverted_y = max(y) - y if len(y) > 0 else y
        self.data_line, = self.ax.plot(x, inverted_y, '-')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(True)
        self.x_data = x
        self.y_data = inverted_y  # Store the inverted data for tooltips
        self.apply_theme(self.current_theme)
        self.draw()

    def on_motion(self, event):
        if event.inaxes != self.ax or not hasattr(self, 'x_data'):
            return
        for ann in self.annotations:
            ann.remove()
        self.annotations.clear()
        
        idx = np.argmin(np.abs(self.x_data - event.xdata))
        x_val = self.x_data[idx]
        y_val = self.y_data[idx]
        
        # Emit the cursor position
        self.cursor_moved.emit(x_val, y_val)
        
        ann = self.ax.annotate(
            f"({x_val:.2f}, {y_val:.2f})",
            xy=(x_val, y_val),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->')
        )
        self.annotations.append(ann)
        self.draw()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        scale_factor = 1/self.zoom_factor if event.button == 'up' else self.zoom_factor
        
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor
        
        self.ax.set_xlim([xdata - new_width/2, xdata + new_width/2])
        self.ax.set_ylim([ydata - new_height/2, ydata + new_height/2])
        self.draw()

    def on_press(self, event):
        if event.inaxes == self.ax and event.button == 1:
            self.pan_start = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.button == 1 and self.pan_start:
            x_start, y_start = self.pan_start
            dx = event.xdata - x_start
            dy = event.ydata - y_start
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
            self.draw()
        self.pan_start = None

    def reset_view(self):
        if hasattr(self, 'x_data'):
            self.ax.set_xlim([min(self.x_data), max(self.x_data)])
            self.ax.set_ylim([min(self.y_data), max(self.y_data)])
            self.draw()

    def apply_theme(self, theme_name):
        self.current_theme = theme_name
        bg_color = '#333333' if theme_name == 'dark' else 'white'
        text_color = 'white' if theme_name == 'dark' else 'black'
        
        self.ax.set_facecolor(bg_color)
        self.fig.set_facecolor(bg_color)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        self.ax.title.set_color(text_color)
        self.ax.tick_params(axis='x', colors=text_color)
        self.ax.tick_params(axis='y', colors=text_color)
        
        for spine in self.ax.spines.values():
            spine.set_color(text_color)
        
        self.draw()

class SerialWorker(QObject):
    data_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    connection_status = Signal(bool)
    acquisition_status = Signal(bool)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_running = False
        self.config = None
        self.integration_time = 0

    def setup_serial(self, port, config):
        try:
            self.serial_port = serial.Serial(port, baudrate=115200, timeout=5)
            self.config = config
            self.connection_status.emit(True)
        except Exception as e:
            self.error_occurred.emit(f"Serial connection error: {str(e)}")
            self.connection_status.emit(False)

    def start_acquisition(self):
        self.is_running = True
        self.acquisition_status.emit(True)

    def stop_acquisition(self):
        self.is_running = False
        self.acquisition_status.emit(False)

    def run(self):
        while True:
            if not self.is_running:
                time.sleep(0.1)
                continue

            try:
                start_time = time.time()
                self.serial_port.reset_input_buffer()
                self.serial_port.write(self.config)
                raw_data = self.serial_port.read(7388)  # 3694 pixels × 2 bytes
                pixel_data = np.frombuffer(raw_data, dtype='<u2')
                self.integration_time = time.time() - start_time
                self.data_ready.emit(pixel_data)
            except Exception as e:
                self.error_occurred.emit(f"Data acquisition error: {str(e)}")
                self.connection_status.emit(False)
                break
            time.sleep(0.1)

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

class IntegrationTimeCalculator(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Integration Time Calculator")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(400, 200)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Input fields
        input_group = QGroupBox("CCD Settings")
        input_layout = QVBoxLayout()
        
        self.sh_spin = QSpinBox()
        self.sh_spin.setRange(1, 10000)
        self.sh_spin.setValue(DEFAULT_SH_PERIOD)
        
        self.icg_spin = QSpinBox()
        self.icg_spin.setRange(1000, 500000)
        self.icg_spin.setValue(DEFAULT_ICG_PERIOD)
        
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self.calculate)
        
        input_layout.addWidget(QLabel("SH Period:"))
        input_layout.addWidget(self.sh_spin)
        input_layout.addWidget(QLabel("ICG Period:"))
        input_layout.addWidget(self.icg_spin)
        input_layout.addWidget(self.calculate_btn)
        input_group.setLayout(input_layout)
        
        # Results display
        self.result_label = QLabel("Results will appear here")
        self.result_label.setWordWrap(True)
        
        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        
        layout.addWidget(input_group)
        layout.addWidget(self.result_label)
        layout.addWidget(self.close_btn)
        self.setLayout(layout)
    
    def calculate(self):
        sh = self.sh_spin.value()
        icg = self.icg_spin.value()
        
        if sh < 20:
            self.result_label.setText("⚠️ SH value too low. Minimum is 20 for 10 µs integration.")
            return

        if icg % sh != 0:
            lower = (icg // sh) * sh
            higher = ((icg // sh) + 1) * sh
            msg = [
                "❌ Invalid ICG! It must be a multiple of SH.",
                f"💡 Suggested valid ICG values for SH = {sh}:",
                f"   ➤ {lower} or {higher}"
            ]
            self.result_label.setText("\n".join(msg))
            return

        # Calculate integration time
        tint = sh / MCLK  # seconds
        tint_ms = tint * 1000  # ms
        tint_us = tint * 1e6  # µs

        n = icg // sh
        mode = "Normal mode" if n == 1 else "Electronic shutter mode"

        # Format results
        results = [
            f"✅ Integration time: {tint_us:.2f} µs ({tint_ms:.3f} ms)",
            f"Shutter mode: {mode} (n = {n})",
            f"SH: {sh}, ICG: {icg}",
            f"ICG/SH ratio: {n}"
        ]
        self.result_label.setText("\n".join(results))

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wavelength Calibration")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(600, 500)
        
        self.calibration_points = []
        self.calibration_coeffs = None
        self.calibration_figure = Figure()
        self.calibration_canvas = FigureCanvas(self.calibration_figure)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Calibration points input
        point_group = QGroupBox("Add Calibration Point")
        point_layout = QHBoxLayout()
        
        self.pixel_input = QSpinBox()
        self.pixel_input.setRange(0, 3693)
        self.pixel_input.setValue(0)
        
        self.wavelength_input = QDoubleSpinBox()
        self.wavelength_input.setRange(200, 1100)
        self.wavelength_input.setValue(532.0)
        self.wavelength_input.setSingleStep(0.1)
        
        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.add_calibration_point)
        
        point_layout.addWidget(QLabel("Pixel:"))
        point_layout.addWidget(self.pixel_input)
        point_layout.addWidget(QLabel("Wavelength (nm):"))
        point_layout.addWidget(self.wavelength_input)
        point_layout.addWidget(self.add_point_btn)
        point_group.setLayout(point_layout)
        
        # Calibration points table
        self.points_table = QLabel("No calibration points added")
        self.points_table.setAlignment(Qt.AlignTop)
        
        # Calibration actions
        action_layout = QHBoxLayout()
        self.calibrate_btn = QPushButton("Perform Calibration")
        self.calibrate_btn.clicked.connect(self.perform_calibration)
        self.clear_btn = QPushButton("Clear Points")
        self.clear_btn.clicked.connect(self.clear_points)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        
        action_layout.addWidget(self.calibrate_btn)
        action_layout.addWidget(self.clear_btn)
        action_layout.addWidget(self.close_btn)
        
        # Add widgets to main layout
        layout.addWidget(point_group)
        layout.addWidget(self.points_table)
        layout.addWidget(self.calibration_canvas)
        layout.addLayout(action_layout)
        
        self.setLayout(layout)
    
    def add_calibration_point(self):
        pixel = self.pixel_input.value()
        wavelength = self.wavelength_input.value()
        self.calibration_points.append((pixel, wavelength))
        self.update_points_table()
        self.update_plot()
    
    def update_points_table(self):
        if not self.calibration_points:
            self.points_table.setText("No calibration points added")
            return
        
        text = "Calibration Points:\nPixel\tWavelength (nm)\n----------------------\n"
        for pixel, wavelength in self.calibration_points:
            text += f"{pixel}\t{wavelength:.2f}\n"
        self.points_table.setText(text)
    
    def clear_points(self):
        self.calibration_points = []
        self.calibration_coeffs = None
        self.update_points_table()
        self.update_plot()
    
    def perform_calibration(self):
        if len(self.calibration_points) < 3:
            QMessageBox.warning(self, "Insufficient Points", 
                              "At least 3 calibration points are required for quadratic fitting.")
            return
        
        pixels = np.array([p[0] for p in self.calibration_points])
        wavelengths = np.array([p[1] for p in self.calibration_points])
        
        # Sort by pixel number
        sort_idx = np.argsort(pixels)
        pixels = pixels[sort_idx]
        wavelengths = wavelengths[sort_idx]
        
        try:
            # Quadratic fitting: λ = a·p² + b·p + c
            coeffs, _ = curve_fit(lambda p, a, b, c: a*p**2 + b*p + c, 
                                 pixels, wavelengths, p0=[1e-5, 0.1, 500])
            
            # Store coefficients and update parent window
            self.calibration_coeffs = coeffs
            if self.parent() is not None:
                main_window = self.parent()
                main_window.calibration_coeffs = coeffs
                if main_window.pixel_data is not None or main_window.loaded_data is not None:
                    main_window.update_plots()
            
            # Calculate fit quality
            predicted = coeffs[0] * pixels**2 + coeffs[1] * pixels + coeffs[2]
            max_error = np.max(np.abs(predicted - wavelengths))
            
            if max_error > 5:
                QMessageBox.warning(self, "Large Calibration Error", 
                                  f"Maximum error is {max_error:.2f} nm. Consider adding more points.")
            
            self.update_plot()
            
            QMessageBox.information(self, "Calibration Success", 
                                   f"Calibration coefficients:\n"
                                   f"a = {coeffs[0]:.6f}\n"
                                   f"b = {coeffs[1]:.6f}\n"
                                   f"c = {coeffs[2]:.6f}\n\n"
                                   f"Max error: {max_error:.2f} nm")
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"Failed to perform calibration: {str(e)}")
    
    def update_plot(self):
        ax = self.calibration_figure.gca()
        ax.clear()
        
        if self.calibration_points:
            pixels = np.array([p[0] for p in self.calibration_points])
            wavelengths = np.array([p[1] for p in self.calibration_points])
            
            # Sort by pixel number
            sort_idx = np.argsort(pixels)
            pixels = pixels[sort_idx]
            wavelengths = wavelengths[sort_idx]
            
            ax.scatter(pixels, wavelengths, color='red', label='Calibration Points')
            
            if self.calibration_coeffs is not None:
                fit_pixels = np.linspace(min(pixels), max(pixels), 100)
                fit_wavelengths = (self.calibration_coeffs[0] * fit_pixels**2 + 
                                  self.calibration_coeffs[1] * fit_pixels + 
                                  self.calibration_coeffs[2])
                ax.plot(fit_pixels, fit_wavelengths, 'b-', label='Quadratic Fit')
        
        ax.set_xlabel('Pixel Number')
        ax.set_ylabel('Wavelength (nm)')
        ax.set_title('Wavelength Calibration')
        ax.grid(True)
        ax.legend()
        
        self.calibration_canvas.draw()
    
    def get_calibration_coeffs(self):
        return self.calibration_coeffs
    
    def get_calibration_points(self):
        return self.calibration_points

class SpectrometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAMAN SPECTROMETER")
        self.setWindowIcon(QIcon.fromTheme("applications-science"))
        self.resize(1200, 800)
        
        # Data storage
        self.pixel_data = None
        self.calibration_coeffs = None
        self.excitation_wavelength = 532.0
        self.integration_time = 0
        self.loaded_data = None
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.serial_worker = SerialWorker()
        self.serial_thread = None
        
        self.init_ui()
        self.apply_theme('light')

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Control panel
        control_panel = QGroupBox("Acquisition Control")
        control_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_ccd)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_ccd)
        self.disconnect_btn.setEnabled(False)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_acquisition)
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.stop_btn.setEnabled(False)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single", "Continuous"])
        
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 15)
        self.avg_spin.setValue(DEFAULT_AVG)
        
        control_layout.addWidget(QLabel("Mode:"))
        control_layout.addWidget(self.mode_combo)
        control_layout.addWidget(QLabel("Averages:"))
        control_layout.addWidget(self.avg_spin)
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.disconnect_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_panel.setLayout(control_layout)
        
        # Settings panel
        settings_panel = QGroupBox("CCD Settings")
        settings_layout = QHBoxLayout()
        
        self.sh_spin = QSpinBox()
        self.sh_spin.setRange(1, 10000)
        self.sh_spin.setValue(DEFAULT_SH_PERIOD)
        
        self.icg_spin = QSpinBox()
        self.icg_spin.setRange(1000, 500000)
        self.icg_spin.setValue(DEFAULT_ICG_PERIOD)
        
        self.integration_label = QLabel("Integration Time: 0 ms")
        
        self.calc_time_btn = QPushButton("Calculate Time")
        self.calc_time_btn.clicked.connect(self.show_time_calculator)
        
        settings_layout.addWidget(QLabel("SH Period:"))
        settings_layout.addWidget(self.sh_spin)
        settings_layout.addWidget(QLabel("ICG Period:"))
        settings_layout.addWidget(self.icg_spin)
        settings_layout.addWidget(self.integration_label)
        settings_layout.addWidget(self.calc_time_btn)
        settings_panel.setLayout(settings_layout)
        
        # Graph tabs
        self.tabs = QTabWidget()
        
        # Pixel vs Intensity tab
        self.pixel_graph = InteractiveGraph()
        self.pixel_toolbar = NavigationToolbar2QT(self.pixel_graph, self)
        pixel_widget = QWidget()
        pixel_layout = QVBoxLayout()
        pixel_layout.addWidget(self.pixel_toolbar)
        pixel_layout.addWidget(self.pixel_graph)
        pixel_widget.setLayout(pixel_layout)
        self.tabs.addTab(pixel_widget, "Pixel vs Intensity")
        
        # Wavelength vs Intensity tab
        self.wavelength_graph = InteractiveGraph()
        self.wavelength_toolbar = NavigationToolbar2QT(self.wavelength_graph, self)
        wavelength_widget = QWidget()
        wavelength_layout = QVBoxLayout()
        wavelength_layout.addWidget(self.wavelength_toolbar)
        wavelength_layout.addWidget(self.wavelength_graph)
        wavelength_widget.setLayout(wavelength_layout)
        self.tabs.addTab(wavelength_widget, "Wavelength vs Intensity")
        
        # Raman Shift vs Intensity tab
        self.raman_graph = InteractiveGraph()
        self.raman_toolbar = NavigationToolbar2QT(self.raman_graph, self)
        raman_widget = QWidget()
        raman_layout = QVBoxLayout()
        raman_layout.addWidget(self.raman_toolbar)
        raman_layout.addWidget(self.raman_graph)
        raman_widget.setLayout(raman_layout)
        self.tabs.addTab(raman_widget, "Raman Shift vs Intensity")
        
        # Bottom panel
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout()
        
        # Add cursor position displays
        bottom_layout.addWidget(QLabel("X:"))
        self.x_display = QLineEdit()
        self.x_display.setReadOnly(True)
        self.x_display.setFixedWidth(120)
        bottom_layout.addWidget(self.x_display)
        
        bottom_layout.addWidget(QLabel("Y:"))
        self.y_display = QLineEdit()
        self.y_display.setReadOnly(True)
        self.y_display.setFixedWidth(120)
        bottom_layout.addWidget(self.y_display)
        
        self.upload_btn = QPushButton("Upload File")
        self.upload_btn.clicked.connect(self.upload_file)
        
        self.save_data_btn = QPushButton("Save Data")
        self.save_data_btn.clicked.connect(self.save_data)
        self.save_data_btn.setEnabled(False)
        
        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self.save_image)
        self.save_image_btn.setEnabled(False)
        
        self.calibrate_btn = QPushButton("Calibrate Wavelength")
        self.calibrate_btn.clicked.connect(self.open_calibration_dialog)
        
        self.excitation_combo = QComboBox()
        self.excitation_combo.addItems([str(wl) for wl in EXCITATION_WAVELENGTHS])
        self.excitation_combo.setCurrentText("532")
        self.excitation_combo.currentTextChanged.connect(self.update_excitation_wavelength)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(THEMES)
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        
        bottom_layout.addWidget(self.upload_btn)
        bottom_layout.addWidget(QLabel("Excitation (nm):"))
        bottom_layout.addWidget(self.excitation_combo)
        bottom_layout.addWidget(QLabel("Theme:"))
        bottom_layout.addWidget(self.theme_combo)
        bottom_layout.addWidget(self.calibrate_btn)
        bottom_layout.addWidget(self.reset_zoom_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.save_data_btn)
        bottom_layout.addWidget(self.save_image_btn)
        bottom_panel.setLayout(bottom_layout)
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(settings_panel)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(bottom_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Connect signals
        self.serial_worker.data_ready.connect(self.update_data)
        self.serial_worker.error_occurred.connect(self.handle_error)
        self.serial_worker.connection_status.connect(self.update_connection_status)
        self.serial_worker.acquisition_status.connect(self.update_acquisition_status)
        
        # Connect graph cursor signals
        self.pixel_graph.cursor_moved.connect(self.update_cursor_display)
        self.wavelength_graph.cursor_moved.connect(self.update_cursor_display)
        self.raman_graph.cursor_moved.connect(self.update_cursor_display)

    def update_cursor_display(self, x_val, y_val):
        """Update the X and Y display boxes with full precision values"""
        # Format to show full decimal without scientific notation
        def format_value(val):
            s = f"{val:.8f}"
            if '.' in s:
                s = s.rstrip('0').rstrip('.')
            return s
        
        self.x_display.setText(format_value(x_val))
        self.y_display.setText(format_value(y_val))

    def show_time_calculator(self):
        dialog = IntegrationTimeCalculator(self)
        dialog.exec()

    # def upload_file(self):
    #     filename, _ = QFileDialog.getOpenFileName(
    #         self, "Open CCD Data File", "", 
    #         "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        
    #     if not filename:
    #         return
        
    #     try:
    #         with open(filename, 'r') as f:
    #             reader = csv.reader(f)
    #             data = []
                
    #             for row in reader:
    #                 if not row or row[0].startswith('#'):
    #                     continue
    #                 try:
    #                     pixel = float(row[0].replace(',', '.'))
    #                     intensity = float(row[-1].replace(',', '.'))
    #                     data.append((pixel, intensity))
    #                 except (ValueError, IndexError):
    #                     continue
                
    #             if not data:
    #                 raise ValueError("No valid data found in file")
                
    #             # Sort by pixel number
    #             data.sort(key=lambda x: x[0])
    #             pixels = np.array([x[0] for x in data])
    #             intensities = np.array([x[1] for x in data])
                
    #             self.loaded_data = (pixels, intensities)
    #             self.pixel_data = intensities
    #             self.update_plots()
    #             self.save_data_btn.setEnabled(True)
    #             self.save_image_btn.setEnabled(True)
    #             self.status_bar.showMessage(f"Loaded data from {os.path.basename(filename)}")
                
    #     except Exception as e:
    #         QMessageBox.critical(self, "Load Error", f"Failed to load file: {str(e)}")
    # def upload_file(self):
    #     filename, _ = QFileDialog.getOpenFileName(
    #         self, "Open CCD Data File", "", 
    #         "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        
    #     if not filename:
    #         return
        
    #     try:
    #         with open(filename, 'r', encoding='utf-8') as f:
    #             lines = f.readlines()
                
    #             data = []
    #             metadata = {}
    #             has_wavelength = False
    #             has_raman_shift = False
    #             excitation_wavelength = None
                
    #             # Parse metadata and detect file format
    #             for line in lines:
    #                 if line.startswith('#'):
    #                     if "Excitation Wavelength:" in line:
    #                         try:
    #                             excitation_wavelength = float(line.split(':')[1].strip().split()[0])
    #                         except (ValueError, IndexError):
    #                             pass
    #                     continue
                    
    #                 if "Wavelength" in line and "Raman" in line:
    #                     has_wavelength = has_raman_shift = True
    #                     break
    #                 elif "Wavelength" in line:
    #                     has_wavelength = True
    #                     break
                
    #             # Process data lines
    #             for line in lines:
    #                 if not line.strip() or line.startswith('#'):
    #                     continue
                    
    #                 parts = [p.strip() for p in line.split(',')]
                    
    #                 try:
    #                     if has_wavelength and has_raman_shift and len(parts) >= 4:
    #                         # Format: Pixel,Wavelength,Raman Shift,Intensity
    #                         pixel = float(parts[0].replace(',', '.'))
    #                         wavelength = float(parts[1].replace(',', '.'))
    #                         raman_shift = float(parts[2].replace(',', '.'))
    #                         intensity = float(parts[3].replace(',', '.'))
    #                         data.append((pixel, wavelength, raman_shift, intensity))
    #                     elif has_wavelength and len(parts) >= 3:
    #                         # Format: Pixel,Wavelength,Intensity
    #                         pixel = float(parts[0].replace(',', '.'))
    #                         wavelength = float(parts[1].replace(',', '.'))
    #                         intensity = float(parts[2].replace(',', '.'))
    #                         data.append((pixel, wavelength, intensity))
    #                     elif len(parts) >= 2:
    #                         # Format: Pixel,Intensity
    #                         pixel = float(parts[0].replace(',', '.'))
    #                         intensity = float(parts[1].replace(',', '.'))
    #                         data.append((pixel, intensity))
    #                 except (ValueError, IndexError):
    #                     continue
                
    #             if not data:
    #                 raise ValueError("No valid data found in file")
                
    #             # Sort by pixel number
    #             data.sort(key=lambda x: x[0])
                
    #             # Extract data based on format
    #             if has_wavelength and has_raman_shift:
    #                 pixels = np.array([x[0] for x in data])
    #                 wavelengths = np.array([x[1] for x in data])
    #                 raman_shifts = np.array([x[2] for x in data])
    #                 intensities = np.array([x[3] for x in data])
                    
    #                 # Store all data
    #                 self.loaded_data = (pixels, intensities)
    #                 self.wavelength_data = wavelengths
    #                 self.raman_shift_data = raman_shifts
    #                 self.pixel_data = intensities
                    
    #             elif has_wavelength:
    #                 pixels = np.array([x[0] for x in data])
    #                 wavelengths = np.array([x[1] for x in data])
    #                 intensities = np.array([x[2] for x in data])
                    
    #                 self.loaded_data = (pixels, intensities)
    #                 self.wavelength_data = wavelengths
    #                 self.pixel_data = intensities
                    
    #                 # Calculate Raman shift if excitation wavelength is available
    #                 if excitation_wavelength is not None:
    #                     self.excitation_wavelength = excitation_wavelength
    #                     self.raman_shift_data = self.calculate_raman_shift(wavelengths)
    #             else:
    #                 pixels = np.array([x[0] for x in data])
    #                 intensities = np.array([x[1] for x in data])
                    
    #                 self.loaded_data = (pixels, intensities)
    #                 self.pixel_data = intensities
                
    #             # Update plots with available data
    #             self.update_plots()
                
    #             # Enable relevant buttons
    #             self.save_data_btn.setEnabled(True)
    #             self.save_image_btn.setEnabled(True)
                
    #             # Update status bar
    #             loaded_info = f"Loaded data from {os.path.basename(filename)}"
    #             if has_wavelength:
    #                 loaded_info += " (with wavelength data)"
    #             if has_raman_shift:
    #                 loaded_info += " (with Raman shift data)"
    #             self.status_bar.showMessage(loaded_info)
                
    #     except Exception as e:
    #         QMessageBox.critical(self, "Load Error", f"Failed to load file: {str(e)}")

    def upload_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CCD Data File", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                data = []
                metadata = {}
                has_wavelength = False
                has_raman_shift = False
                excitation_wavelength = None
                
                # Parse metadata and detect file format
                for line in lines:
                    if line.startswith('#'):
                        if "Excitation Wavelength:" in line:
                            try:
                                excitation_wavelength = float(line.split(':')[1].strip().split()[0])
                                self.excitation_wavelength = excitation_wavelength
                                self.excitation_combo.setCurrentText(str(excitation_wavelength))
                            except (ValueError, IndexError):
                                pass
                        if "Wavelength Calibration:" in line:
                            # Check if calibration coefficients are available
                            self.calibration_coeffs = [0.0, 0.0, 0.0]  # Initialize with dummy values
                        continue
                    
                    # Check for header line
                    if "Wavelength" in line and "Raman" in line:
                        has_wavelength = has_raman_shift = True
                        continue
                    elif "Wavelength" in line:
                        has_wavelength = True
                        continue
                
                # Process data lines
                for line in lines:
                    if not line.strip() or line.startswith('#'):
                        continue
                    
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                    
                    try:
                        if has_wavelength and has_raman_shift and len(parts) >= 4:
                            # Format: Pixel,Wavelength,Raman Shift,Intensity
                            pixel = float(parts[0].replace(',', '.'))
                            wavelength = float(parts[1].replace(',', '.'))
                            raman_shift = float(parts[2].replace(',', '.'))
                            intensity = float(parts[3].replace(',', '.'))
                            data.append((pixel, wavelength, raman_shift, intensity))
                        elif has_wavelength and len(parts) >= 3:
                            # Format: Pixel,Wavelength,Intensity
                            pixel = float(parts[0].replace(',', '.'))
                            wavelength = float(parts[1].replace(',', '.'))
                            intensity = float(parts[2].replace(',', '.'))
                            data.append((pixel, wavelength, intensity))
                        elif len(parts) >= 2:
                            # Format: Pixel,Intensity
                            pixel = float(parts[0].replace(',', '.'))
                            intensity = float(parts[1].replace(',', '.'))
                            data.append((pixel, intensity))
                    except (ValueError, IndexError):
                        continue
                
                if not data:
                    raise ValueError("No valid data found in file")
                
                # Sort by pixel number
                data.sort(key=lambda x: x[0])
                
                # Extract data based on format
                if has_wavelength and has_raman_shift:
                    pixels = np.array([x[0] for x in data])
                    wavelengths = np.array([x[1] for x in data])
                    raman_shifts = np.array([x[2] for x in data])
                    intensities = np.array([x[3] for x in data])
                    
                    # Store all data
                    self.loaded_data = (pixels, intensities)
                    self.wavelength_data = wavelengths
                    self.raman_shift_data = raman_shifts
                    self.pixel_data = intensities
                    
                    # Set calibration flag to enable all graphs
                    self.calibration_coeffs = [0.0, 0.0, 0.0]  # Dummy values since we have actual wavelengths
                    
                elif has_wavelength:
                    pixels = np.array([x[0] for x in data])
                    wavelengths = np.array([x[1] for x in data])
                    intensities = np.array([x[2] for x in data])
                    
                    self.loaded_data = (pixels, intensities)
                    self.wavelength_data = wavelengths
                    self.pixel_data = intensities
                    
                    # Calculate Raman shift if excitation wavelength is available
                    if excitation_wavelength is not None:
                        self.excitation_wavelength = excitation_wavelength
                        self.raman_shift_data = self.calculate_raman_shift(wavelengths)
                else:
                    pixels = np.array([x[0] for x in data])
                    intensities = np.array([x[1] for x in data])
                    
                    self.loaded_data = (pixels, intensities)
                    self.pixel_data = intensities
                
                # Update plots with available data
                self.update_plots()
                
                # Enable relevant buttons
                self.save_data_btn.setEnabled(True)
                self.save_image_btn.setEnabled(True)
                
                # Update status bar
                loaded_info = f"Loaded data from {os.path.basename(filename)}"
                if has_wavelength:
                    loaded_info += " (with wavelength data)"
                if has_raman_shift:
                    loaded_info += " (with Raman shift data)"
                self.status_bar.showMessage(loaded_info)
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file: {str(e)}")

    def connect_ccd(self):
        ports = [p.device for p in serial.tools.list_ports.comports()
                if "STM" in p.description or "Virtual COM Port" in p.description]
        
        if not ports:
            QMessageBox.critical(self, "Error", "No STM32 device found.")
            return
        
        port = ports[0]
        sh = self.sh_spin.value()
        icg = self.icg_spin.value()
        avg = self.avg_spin.value()
        mode = 1 if self.mode_combo.currentText() == "Continuous" else 0
        
        config = bytearray([69, 82])  # 'ER' header
        config += sh.to_bytes(4, 'big')
        config += icg.to_bytes(4, 'big')
        config += bytes([mode, avg])
        
        self.serial_worker.setup_serial(port, config)
        
        if not self.serial_thread or not self.serial_thread.isRunning():
            self.serial_thread = QThread()
            self.serial_worker.moveToThread(self.serial_thread)
            self.serial_thread.started.connect(self.serial_worker.run)
            self.serial_thread.start()
        
        self.status_bar.showMessage(f"Connected to {port}")

    def disconnect_ccd(self):
        self.stop_acquisition()
        self.serial_worker.close()
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.quit()
            self.serial_thread.wait()
        self.status_bar.showMessage("Disconnected")

    def start_acquisition(self):
        self.serial_worker.start_acquisition()
        self.status_bar.showMessage("Acquisition started...")

    def stop_acquisition(self):
        self.serial_worker.stop_acquisition()
        self.status_bar.showMessage("Acquisition stopped")

    def update_connection_status(self, connected):
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.start_btn.setEnabled(connected)

    def update_acquisition_status(self, running):
        self.stop_btn.setEnabled(running)
        self.start_btn.setEnabled(not running and self.disconnect_btn.isEnabled())

    def update_data(self, data):
        self.pixel_data = data
        self.loaded_data = None
        self.integration_time = self.serial_worker.integration_time
        self.integration_label.setText(f"Integration Time: {self.integration_time*1000:.1f} ms")
        self.update_plots()
        self.save_data_btn.setEnabled(True)
        self.save_image_btn.setEnabled(True)
        
        if self.mode_combo.currentText() == "Single":
            self.stop_acquisition()

    def pixel_to_wavelength(self, pixels):
        """Convert pixel numbers to wavelengths using calibration coefficients"""
        if self.calibration_coeffs is None:
            return None
        return (self.calibration_coeffs[0] * pixels**2 + 
                self.calibration_coeffs[1] * pixels + 
                self.calibration_coeffs[2])

    def calculate_raman_shift(self, wavelengths):
        """Calculate Raman shift from wavelengths"""
        if wavelengths is None:
            return None
        return 1e7 * (1/self.excitation_wavelength - 1/wavelengths)

    def update_plots(self):
        if self.pixel_data is None and self.loaded_data is None:
            return
        
        if self.loaded_data:
            pixels, intensities = self.loaded_data
        else:
            pixels = np.arange(len(self.pixel_data))
            intensities = self.pixel_data
        
        # Pixel vs Intensity (always available)
        self.pixel_graph.plot_data(
            pixels, intensities,
            'Pixel Number', 'Intensity (counts)',
            f'Pixel vs Intensity{" (Loaded Data)" if self.loaded_data else ""}'
        )
        
        # Wavelength vs Intensity (only if calibrated)
        wavelengths = self.pixel_to_wavelength(pixels)
        if wavelengths is not None:
            self.wavelength_graph.plot_data(
                wavelengths, intensities,
                'Wavelength (nm)', 'Intensity (counts)',
                f'Wavelength vs Intensity (Excitation: {self.excitation_wavelength} nm)'
            )
            
            # Raman Shift vs Intensity
            raman_shift = self.calculate_raman_shift(wavelengths)
            self.raman_graph.plot_data(
                raman_shift, intensities,
                'Raman Shift (cm⁻¹)', 'Intensity (counts)',
                f'Raman Shift vs Intensity (Excitation: {self.excitation_wavelength} nm)'
            )
        else:
            # Clear plots if no calibration
            self.wavelength_graph.plot_data([], [], 'Wavelength (nm)', 'Intensity (counts)', 'Wavelength vs Intensity (No calibration)')
            self.raman_graph.plot_data([], [], 'Raman Shift (cm⁻¹)', 'Intensity (counts)', 'Raman Shift vs Intensity (No calibration)')

    def reset_zoom(self):
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            self.pixel_graph.reset_view()
        elif current_tab == 1:
            self.wavelength_graph.reset_view()
        elif current_tab == 2:
            self.raman_graph.reset_view()

    def open_calibration_dialog(self):
        dialog = CalibrationDialog(self)
        if dialog.exec() == QDialog.Accepted:
            new_coeffs = dialog.get_calibration_coeffs()
            if new_coeffs is not None:
                self.calibration_coeffs = new_coeffs
                if self.pixel_data is not None or self.loaded_data is not None:
                    self.update_plots()

    # def save_data(self):
    #     if self.pixel_data is None and self.loaded_data is None:
    #         return
        
    #     filename, _ = QFileDialog.getSaveFileName(
    #         self, "Save Data", "", "CSV Files (*.csv);;Text Files (*.txt)")
        
    #     if not filename:
    #         return
        
    #     try:
    #         if self.loaded_data:
    #             pixels, intensities = self.loaded_data
    #         else:
    #             pixels = np.arange(len(self.pixel_data))
    #             intensities = self.pixel_data
            
    #         metadata = [
    #             f"# CCD Spectrometer Data - {datetime.now().isoformat()}",
    #             f"# Source: {'Uploaded File' if self.loaded_data else 'CCD Acquisition'}",
    #             f"# Excitation Wavelength: {self.excitation_wavelength} nm"
    #         ]
            
    #         if self.calibration_coeffs is not None:
    #             metadata.extend([
    #                 "# Wavelength Calibration:",
    #                 f"# a = {self.calibration_coeffs[0]:.6f}",
    #                 f"# b = {self.calibration_coeffs[1]:.6f}",
    #                 f"# c = {self.calibration_coeffs[2]:.6f}"
    #             ])
            
    #         with open(filename, 'w', newline='') as f:
    #             writer = csv.writer(f, delimiter='\t')
    #             for line in metadata:
    #                 writer.writerow([line])
                
    #             if self.calibration_coeffs is not None:
    #                 writer.writerow(["Pixel", "Wavelength (nm)", "Raman Shift (cm⁻¹)", "Intensity"])
    #                 wavelengths = self.pixel_to_wavelength(pixels)
    #                 raman_shift = self.calculate_raman_shift(wavelengths)
                    
    #                 for px, wl, rs, intensity in zip(pixels, wavelengths, raman_shift, intensities):
    #                     writer.writerow([px, wl, rs, intensity])
    #             else:
    #                 writer.writerow(["Pixel", "Intensity"])
    #                 for px, intensity in zip(pixels, intensities):
    #                     writer.writerow([px, intensity])
            
    #         self.status_bar.showMessage(f"Data saved to {filename}")
    #     except Exception as e:
    #         QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
    # def save_data(self):
    #     if self.pixel_data is None and self.loaded_data is None:
    #         return
        
    #     filename, _ = QFileDialog.getSaveFileName(
    #         self, "Save Data", "", "CSV Files (*.csv);;Text Files (*.txt)")
        
    #     if not filename:
    #         return
        
    #     try:
    #         if self.loaded_data:
    #             pixels, intensities = self.loaded_data
    #         else:
    #             pixels = np.arange(len(self.pixel_data))
    #             intensities = self.pixel_data
            
    #         # Prepare metadata lines
    #         metadata = [
    #             f"# CCD Spectrometer Data - {datetime.now().isoformat()}",
    #             f"# Source: {'Uploaded File' if self.loaded_data else 'CCD Acquisition'}",
    #             f"# Excitation Wavelength: {self.excitation_wavelength} nm"
    #         ]
            
    #         if self.calibration_coeffs is not None:
    #             metadata.extend([
    #                 "# Wavelength Calibration:",
    #                 f"# a = {self.calibration_coeffs[0]:.6f}",
    #                 f"# b = {self.calibration_coeffs[1]:.6f}",
    #                 f"# c = {self.calibration_coeffs[2]:.6f}"
    #             ])
            
    #         with open(filename, 'w', newline='') as f:
    #             # Write metadata as comments
    #             for line in metadata:
    #                 f.write(line + '\n')
                
    #             if self.calibration_coeffs is not None:
    #                 # Write header with proper spacing
    #                 header = "Pixel,Wavelength (nm),Raman Shift (cm⁻¹),Intensity"
    #                 f.write(header + '\n')
                    
    #                 wavelengths = self.pixel_to_wavelength(pixels)
    #                 raman_shift = self.calculate_raman_shift(wavelengths)
                    
    #                 for px, wl, rs, intensity in zip(pixels, wavelengths, raman_shift, intensities):
    #                     line = f"{px},{wl:.2f},{rs:.2f},{intensity}"
    #                     f.write(line + '\n')
    #             else:
    #                 # Write header with proper spacing
    #                 header = "Pixel,Intensity"
    #                 f.write(header + '\n')
                    
    #                 for px, intensity in zip(pixels, intensities):
    #                     line = f"{px},{intensity}"
    #                     f.write(line + '\n')
            
    #         self.status_bar.showMessage(f"Data saved to {filename}")
    #     except Exception as e:
    #         QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")

    def save_data(self):
        if self.pixel_data is None and self.loaded_data is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "", "CSV Files (*.csv);;Text Files (*.txt)")
        
        if not filename:
            return
        
        try:
            if self.loaded_data:
                pixels, intensities = self.loaded_data
            else:
                pixels = np.arange(len(self.pixel_data))
                intensities = self.pixel_data
            
            # Prepare metadata lines
            metadata = [
                f"# CCD Spectrometer Data - {datetime.now().isoformat()}",
                f"# Source: {'Uploaded File' if self.loaded_data else 'CCD Acquisition'}",
                f"# Excitation Wavelength: {self.excitation_wavelength} nm"
            ]
            
            if self.calibration_coeffs is not None:
                metadata.extend([
                    "# Wavelength Calibration:",
                    f"# a = {self.calibration_coeffs[0]:.6f}",
                    f"# b = {self.calibration_coeffs[1]:.6f}",
                    f"# c = {self.calibration_coeffs[2]:.6f}"
                ])
            
            # Open file with UTF-8 encoding to handle special characters
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                # Write metadata as comments
                for line in metadata:
                    f.write(line + '\n')
                
                if self.calibration_coeffs is not None:
                    # Write header with proper spacing
                    header = "Pixel,Wavelength (nm),Raman Shift (cm⁻¹),Intensity"
                    f.write(header + '\n')
                    
                    wavelengths = self.pixel_to_wavelength(pixels)
                    raman_shift = self.calculate_raman_shift(wavelengths)
                    
                    for px, wl, rs, intensity in zip(pixels, wavelengths, raman_shift, intensities):
                        line = f"{px},{wl:.2f},{rs:.2f},{intensity}"
                        f.write(line + '\n')
                else:
                    # Write header with proper spacing
                    header = "Pixel,Intensity"
                    f.write(header + '\n')
                    
                    for px, intensity in zip(pixels, intensities):
                        line = f"{px},{intensity}"
                        f.write(line + '\n')
            
            self.status_bar.showMessage(f"Data saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
    def save_image(self):
        if self.pixel_data is None and self.loaded_data is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)")
        
        if not filename:
            return
        
        try:
            current_tab = self.tabs.currentIndex()
            if current_tab == 0:
                fig = self.pixel_graph.fig
            elif current_tab == 1:
                fig = self.wavelength_graph.fig
            elif current_tab == 2:
                fig = self.raman_graph.fig
            else:
                return
            
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.status_bar.showMessage(f"Image saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image: {str(e)}")

    def update_excitation_wavelength(self, wavelength_str):
        try:
            self.excitation_wavelength = float(wavelength_str)
            if (self.pixel_data is not None or self.loaded_data is not None) and self.calibration_coeffs is not None:
                self.update_plots()
        except ValueError:
            pass

    def apply_theme(self, theme_name):
        if theme_name == 'dark':
            plt.style.use('dark_background')
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #333333;
                    color: #ffffff;
                }
                QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #555555;
                    color: #ffffff;
                    border: 1px solid #777777;
                }
                QGroupBox {
                    border: 1px solid #777777;
                    border-radius: 5px;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
                QLineEdit {
                    background-color: #555555;
                    color: #ffffff;
                    border: 1px solid #777777;
                }
            """)
        else:
            plt.style.use('default')
            self.setStyleSheet("""
                QLineEdit {
                    background-color: white;
                    color: black;
                    border: 1px solid #777777;
                }
            """)
        
        self.pixel_graph.apply_theme(theme_name)
        self.wavelength_graph.apply_theme(theme_name)
        self.raman_graph.apply_theme(theme_name)
        self.update_plots()

    def handle_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.disconnect_ccd()

    def closeEvent(self, event):
        self.disconnect_ccd()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SpectrometerApp()
    window.show()
    sys.exit(app.exec())